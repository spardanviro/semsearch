"""Incremental index updates: only changed files are re-chunked, only new chunks re-embedded."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import blake3
import numpy as np

from semsearch.config import Settings
from semsearch.embed.base import Embedder
from semsearch.index import keywords
from semsearch.index.chunkers import Chunk, chunk_text
from semsearch.index.chunkers.base import chunk_body
from semsearch.index.scanner import FileInfo, read_text
from semsearch.index.store import FileRow, NewChunk, Store
from semsearch.index.vectors import VectorCache
from semsearch.paths import to_abs

log = logging.getLogger(__name__)
UNREADABLE = "!unreadable"
ProgressFn = Callable[[int, int], None]


def text_hash(text: str) -> str:
    return blake3.blake3(text.encode("utf-8")).hexdigest()


@dataclass
class UpdateStats:
    files_indexed: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    files_failed: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    seconds: float = 0.0

    def merge(self, other: UpdateStats) -> None:
        for name in vars(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class _Pending:
    info: FileInfo
    content_hash: str
    chunks: list[Chunk]
    hashes: list[str] = field(default_factory=list)


def diff(known: dict[str, FileRow], current: dict[str, FileInfo]) -> tuple[list[FileInfo], list[str]]:
    changed = [f for p, f in current.items()
               if (row := known.get(p)) is None
               or row.size != f.size or row.mtime_ns != f.mtime_ns]
    deleted = [p for p in known if p not in current]
    return changed, deleted


class Indexer:
    def __init__(self, root: Path, store: Store, cache: VectorCache, embedder: Embedder,
                 settings: Settings, db_lock: threading.Lock) -> None:
        self.root = root
        self.store = store
        self.cache = cache
        self.embedder = embedder
        self.settings = settings
        self.db_lock = db_lock

    def known_files(self) -> dict[str, FileRow]:
        with self.db_lock:
            return self.store.files()

    def apply(self, changed: Sequence[FileInfo], deleted: Sequence[str],
              progress: ProgressFn | None = None) -> UpdateStats:
        t0 = time.perf_counter()
        stats = UpdateStats()
        if deleted:
            self._delete(deleted)
            stats.files_deleted = len(deleted)
        known = self.known_files()
        pending: list[_Pending] = []
        pending_chunks = 0
        for done, info in enumerate(changed, 1):
            item = self._prepare(info, known.get(info.path), stats)
            if item is not None:
                pending.append(item)
                pending_chunks += len(item.chunks)
            if pending_chunks >= self.settings.commit_every_chunks:
                self._flush(pending, stats)
                pending, pending_chunks = [], 0
            if progress:
                progress(done, len(changed))
        if pending:
            self._flush(pending, stats)
        stats.seconds = round(time.perf_counter() - t0, 3)
        return stats

    def _delete(self, paths: Sequence[str]) -> None:
        removed: list[int] = []
        with self.db_lock:
            with self.store.transaction():
                for p in paths:
                    removed.extend(self.store.delete_file(p))
            self.cache.remove(removed)

    def _prepare(self, info: FileInfo, row: FileRow | None, stats: UpdateStats) -> _Pending | None:
        try:
            text = read_text(to_abs(self.root, info.path), self.settings.max_file_bytes)
        except OSError as exc:
            # often transient on Windows (antivirus/editor lock): record nothing, retry next scan
            log.warning("cannot read %s: %s", info.path, exc)
            stats.files_failed += 1
            return None
        if text is None:  # binary or grew too large: remember it so it is not retried
            stats.files_failed += 1
            return _Pending(info, UNREADABLE, [])
        content_hash = text_hash(text)
        if row is not None and row.content_hash == content_hash:
            with self.db_lock:
                self.store.touch_file(row.id, info.size, info.mtime_ns)
            stats.files_unchanged += 1
            return None
        chunks = self._chunk(info, text)
        if chunks is None:
            stats.files_failed += 1
            return _Pending(info, UNREADABLE, [])
        return _Pending(info, content_hash, chunks, [text_hash(c.text) for c in chunks])

    def _chunk(self, info: FileInfo, text: str) -> list[Chunk] | None:
        """One poisonous file must never block the rest of the index: fall back to plain
        line windows, and give up on the file only if even that fails."""
        count = self.embedder.count_tokens
        try:
            return chunk_text(info.path, text, info.modality, info.lang, count, self.settings)
        except Exception:
            log.exception("chunking %s failed, falling back to line windows", info.path)
        try:
            return chunk_text(info.path, text, "text", "text", count, self.settings)
        except Exception:
            log.exception("line-window fallback for %s failed, skipping file", info.path)
            return None

    def _flush(self, pending: list[_Pending], stats: UpdateStats) -> None:
        all_hashes = [h for p in pending for h in p.hashes]
        with self.db_lock:
            vectors = self.store.embeddings_by_hash(all_hashes)
        missing: dict[str, str] = {}
        for p in pending:
            for h, c in zip(p.hashes, p.chunks, strict=True):
                if h not in vectors:
                    missing.setdefault(h, c.text)
        stats.chunks_reused += sum(1 for h in all_hashes if h in vectors)
        if missing:
            embedded = self.embedder.encode_documents(list(missing.values()))
            vectors.update(zip(missing.keys(), embedded, strict=True))
            stats.chunks_embedded += len(missing)

        removed: list[int] = []
        added: list[tuple[int, str, str, np.ndarray]] = []
        now = time.time()
        with self.db_lock, self.store.transaction():
            for p in pending:
                fid = self.store.upsert_file(p.info.path, p.info.size, p.info.mtime_ns,
                                             p.content_hash, p.info.modality, p.info.lang, now)
                removed.extend(self.store.delete_chunks_of_file(fid))
                rows = [NewChunk(h, p.info.modality, c.start_line, c.end_line, c.symbol, c.kind,
                                 c.preview, c.tokens, vectors[h],
                                 keywords.document(f"{p.info.path} {c.symbol}"),
                                 keywords.document(chunk_body(c.text)))
                        for h, c in zip(p.hashes, p.chunks, strict=True)]
                ids = self.store.insert_chunks(fid, rows)
                added.extend((cid, p.info.path, p.info.modality, r.embedding)
                             for cid, r in zip(ids, rows, strict=True))
                if p.content_hash != UNREADABLE:
                    stats.files_indexed += 1
            # still under db_lock: searches never see the DB and the cache disagree
            self.cache.remove(removed)
            self.cache.add(added)
