"""SQLite storage for one project's index. Not thread-safe: callers hold the project lock."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  modality TEXT NOT NULL,
  lang TEXT NOT NULL,
  indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,  -- ids are never reused, so a stale cache row cannot
                                         -- be paired with another chunk's metadata
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_hash TEXT NOT NULL,
  modality TEXT NOT NULL,
  start_line INTEGER, end_line INTEGER,
  page INTEGER, t_start REAL, t_end REAL,
  symbol TEXT, kind TEXT,
  preview TEXT NOT NULL,
  tokens INTEGER NOT NULL,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);
-- keyword index; rowid = chunks.id. Contentless: only terms are stored, text stays in chunks.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  meta, body, content='', contentless_delete=1,
  tokenize="unicode61 remove_diacritics 2 tokenchars '_'"
);
"""


@dataclass(frozen=True)
class FileRow:
    id: int
    path: str
    size: int
    mtime_ns: int
    content_hash: str


@dataclass(frozen=True)
class NewChunk:
    chunk_hash: str
    modality: str
    start_line: int | None
    end_line: int | None
    symbol: str
    kind: str
    preview: str
    tokens: int
    embedding: np.ndarray
    keywords_meta: str = ""    # keywords.document(path + symbol)
    keywords_body: str = ""    # keywords.document(chunk body)


@dataclass(frozen=True)
class ChunkHit:
    id: int
    path: str
    modality: str
    start_line: int | None
    end_line: int | None
    page: int | None
    t_start: float | None
    t_end: float | None
    symbol: str
    kind: str
    preview: str


def to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # --- meta -----------------------------------------------------------------------------
    def meta(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT key, value FROM meta").fetchall())

    def ensure_meta(self, expected: dict[str, str]) -> bool:
        """Wipe the index when model/schema/chunker settings changed. True if it was wiped."""
        current = self.meta()
        if all(current.get(k) == v for k, v in expected.items()):
            return False
        if current.get("schema_version") != expected.get("schema_version"):
            # table layout may differ: recreate instead of just deleting rows
            self.db.executescript("DROP TABLE IF EXISTS chunks_fts; DROP TABLE IF EXISTS chunks; "
                                  "DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS meta;")
            self.db.executescript(SCHEMA)
        with self.transaction():
            self.db.execute("DELETE FROM chunks_fts")
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM files")
            self.db.execute("DELETE FROM meta")
            self.db.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", expected.items())
        return bool(current)

    def clear(self) -> None:
        with self.transaction():
            self.db.execute("DELETE FROM chunks_fts")
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM files")

    # --- transactions ---------------------------------------------------------------------
    def transaction(self) -> _Tx:
        return _Tx(self.db)

    # --- files ----------------------------------------------------------------------------
    def files(self) -> dict[str, FileRow]:
        rows = self.db.execute("SELECT id, path, size, mtime_ns, content_hash FROM files")
        return {r[1]: FileRow(*r) for r in rows}

    def touch_file(self, file_id: int, size: int, mtime_ns: int) -> None:
        self.db.execute("UPDATE files SET size = ?, mtime_ns = ? WHERE id = ?",
                        (size, mtime_ns, file_id))

    def upsert_file(self, path: str, size: int, mtime_ns: int, content_hash: str,
                    modality: str, lang: str, now: float) -> int:
        row = self.db.execute(
            "INSERT INTO files(path, size, mtime_ns, content_hash, modality, lang, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(path) DO UPDATE SET size = excluded.size, "
            "mtime_ns = excluded.mtime_ns, content_hash = excluded.content_hash, "
            "modality = excluded.modality, lang = excluded.lang, indexed_at = excluded.indexed_at "
            "RETURNING id", (path, size, mtime_ns, content_hash, modality, lang, now)).fetchone()
        return int(row[0])

    def delete_file(self, path: str) -> list[int]:
        row = self.db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return []
        removed = self.delete_chunks_of_file(row[0])
        self.db.execute("DELETE FROM files WHERE id = ?", (row[0],))
        return removed

    # --- chunks ---------------------------------------------------------------------------
    def chunk_ids_of_file(self, file_id: int) -> list[int]:
        rows = self.db.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))
        return [r[0] for r in rows]

    def delete_chunks_of_file(self, file_id: int) -> list[int]:
        ids = self.chunk_ids_of_file(file_id)
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" * len(part))
            self.db.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({marks})", part)
        self.db.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        return ids

    def embeddings_by_hash(self, hashes: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(hashes))
        for i in range(0, len(unique), 500):
            part = unique[i:i + 500]
            marks = ",".join("?" * len(part))
            for h, blob in self.db.execute(
                    f"SELECT chunk_hash, embedding FROM chunks WHERE chunk_hash IN ({marks})", part):
                out.setdefault(h, from_blob(blob))
        return out

    def insert_chunks(self, file_id: int, chunks: Sequence[NewChunk]) -> list[int]:
        ids = []
        for c in chunks:
            row = self.db.execute(
                "INSERT INTO chunks(file_id, chunk_hash, modality, start_line, end_line, symbol, "
                "kind, preview, tokens, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "RETURNING id",
                (file_id, c.chunk_hash, c.modality, c.start_line, c.end_line, c.symbol, c.kind,
                 c.preview, c.tokens, to_blob(c.embedding))).fetchone()
            ids.append(int(row[0]))
            if c.keywords_meta or c.keywords_body:
                self.db.execute("INSERT INTO chunks_fts(rowid, meta, body) VALUES (?, ?, ?)",
                                (row[0], c.keywords_meta, c.keywords_body))
        return ids

    def keyword_search(self, match: str, limit: int,
                       weights: tuple[float, float]) -> list[tuple[int, float]]:
        """-> [(chunk id, bm25 score)], best first; scores are positive (higher = better)."""
        rows = self.db.execute(
            "SELECT rowid, -bm25(chunks_fts, ?, ?) AS s FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY s DESC LIMIT ?", (weights[0], weights[1], match, limit))
        return [(int(cid), float(score)) for cid, score in rows]

    def iter_vectors(self) -> Iterator[tuple[int, str, str, np.ndarray]]:
        rows = self.db.execute("SELECT c.id, f.path, c.modality, c.embedding FROM chunks c "
                               "JOIN files f ON f.id = c.file_id")
        for cid, path, modality, blob in rows:
            yield cid, path, modality, from_blob(blob)

    def get_chunks(self, ids: Sequence[int]) -> dict[int, ChunkHit]:
        out: dict[int, ChunkHit] = {}
        for i in range(0, len(ids), 500):
            part = list(ids[i:i + 500])
            marks = ",".join("?" * len(part))
            rows = self.db.execute(
                "SELECT c.id, f.path, c.modality, c.start_line, c.end_line, c.page, c.t_start, "
                f"c.t_end, c.symbol, c.kind, c.preview FROM chunks c JOIN files f "
                f"ON f.id = c.file_id WHERE c.id IN ({marks})", part)
            for r in rows:
                out[r[0]] = ChunkHit(*r)
        return out

    def counts(self) -> tuple[int, int]:
        files = self.db.execute("SELECT count(*) FROM files").fetchone()[0]
        chunks = self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return int(files), int(chunks)


class _Tx:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def __enter__(self) -> sqlite3.Connection:
        self.db.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self.db.execute("ROLLBACK")
            return
        try:
            self.db.execute("COMMIT")
        except sqlite3.Error:
            # e.g. disk full: never leave the shared connection inside an open transaction
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
