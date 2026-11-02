"""Incremental index updates: only changed files are re-chunked, only new chunks re-embedded."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from semsearch.config import MEDIA_KINDS, MEDIA_VERSION, Settings
from semsearch.embed.base import Embedder
from semsearch.index import keywords, media
from semsearch.index.chunkers import Chunk, chunk_text
from semsearch.index.chunkers.base import chunk_body
from semsearch.index.media import MediaChunk
from semsearch.index.scanner import FileInfo, read_text
from semsearch.index.store import FileRow, NewChunk, Store
from semsearch.index.vectors import VectorCache
from semsearch.paths import to_abs

log = logging.getLogger(__name__)
UNREADABLE = "!unreadable"
RETRY_HASH = "!retry"      # some chunks failed transiently: prepare the file again next scan
RETRY_MTIME = 0
ProgressFn = Callable[[int, int], None]
HASH_BLOCK = 1 << 20
MEDIA_BATCH = {"image": 8, "pdf": 2, "audio": 4, "video": 1}   # inputs per encode call
MEDIA_KIND_LABEL = {"image": "image", "pdf": "page", "audio": "segment", "video": "segment"}


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


@dataclass
class _Pending:
    info: FileInfo
    content_hash: str
    chunks: list[Chunk]
    hashes: list[str] = field(default_factory=list)   # one per chunk, or per media chunk
    media: list[MediaChunk] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.media) or len(self.chunks)


def _is_transient(exc: BaseException) -> bool:
    """Out-of-memory (GPU busy elsewhere) may succeed later; other model errors will not."""
    return isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError"


def file_hash(path: Path) -> str:
    h = blake3.blake3()
    with open(path, "rb") as fh:
        while block := fh.read(HASH_BLOCK):
            h.update(block)
    return h.hexdigest()


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
                pending_chunks += item.size
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
        if info.modality in MEDIA_KINDS:
            return self._prepare_media(info, row, stats)
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

    def _prepare_media(self, info: FileInfo, row: FileRow | None,
                       stats: UpdateStats) -> _Pending | None:
        path = to_abs(self.root, info.path)
        try:
            content_hash = file_hash(path)
        except OSError as exc:      # locked by another program: retry next scan
            log.warning("cannot read %s: %s", info.path, exc)
            stats.files_failed += 1
            return None
        if row is not None and row.content_hash == content_hash:
            with self.db_lock:
                self.store.touch_file(row.id, info.size, info.mtime_ns)
            stats.files_unchanged += 1
            return None
        try:
            chunks = media.segment(path, info.modality, self.settings)
        except Exception as exc:    # corrupt or unsupported file: remember, do not retry
            log.warning("cannot read %s %s: %s", info.modality, info.path, exc)
            stats.files_failed += 1
            return _Pending(info, UNREADABLE, [])
        with_audio = "audio" in self.settings.media
        hashes = [text_hash(f"{MEDIA_VERSION}|{with_audio}|{content_hash}|{c.locator}")
                  for c in chunks]
        return _Pending(info, content_hash, [], hashes, chunks)

    def _embed_media(self, wanted: dict[str, tuple[Path, MediaChunk]]
                     ) -> tuple[dict[str, np.ndarray], set[str]]:
        """Decode and embed media chunks a few at a time. -> (vectors, hashes that failed for
        a transient reason). A chunk that cannot be decoded is skipped for good (that part of
        the file is broken); its file keeps the other chunks."""
        with_audio = "audio" in self.settings.media
        out: dict[str, np.ndarray] = {}
        transient: set[str] = set()
        by_kind: dict[str, list[tuple[str, Path, MediaChunk]]] = {}
        for h, (path, chunk) in wanted.items():
            by_kind.setdefault(chunk.kind, []).append((h, path, chunk))
        try:
            for kind, items in by_kind.items():
                size = MEDIA_BATCH.get(kind, 1)
                for i in range(0, len(items), size):
                    batch: list[tuple[str, dict[str, Any]]] = []
                    for h, path, chunk in items[i:i + size]:
                        try:
                            batch.append((h, media.load(path, chunk, with_audio)))
                        except Exception as exc:
                            log.warning("cannot decode %s %s: %s", path.name, chunk.symbol, exc)
                    out.update(self._encode_media_batch(batch, wanted, transient))
        finally:
            media.release()
        return out, transient

    def _encode_media_batch(self, batch: list[tuple[str, dict[str, Any]]],
                            wanted: dict[str, tuple[Path, MediaChunk]],
                            transient: set[str]) -> dict[str, np.ndarray]:
        """A model error on one input must not sink the whole update: retry one by one and
        leave out what still fails. Out-of-memory counts as transient (retried next scan)."""
        if not batch:
            return {}
        try:
            vecs = self.embedder.encode_media([item for _, item in batch])
            return dict(zip([h for h, _ in batch], vecs, strict=True))
        except Exception as exc:
            if len(batch) > 1:
                log.warning("media batch of %d failed (%s); retrying one by one", len(batch), exc)
                out: dict[str, np.ndarray] = {}
                for one in batch:
                    out.update(self._encode_media_batch([one], wanted, transient))
                return out
            path, chunk = wanted[batch[0][0]]
            log.warning("cannot embed %s %s: %s", path.name, chunk.symbol, exc)
            if _is_transient(exc):
                transient.add(batch[0][0])
            return {}

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
        missing_media: dict[str, tuple[Path, MediaChunk]] = {}
        for p in pending:
            if p.media:
                for h, m in zip(p.hashes, p.media, strict=True):
                    if h not in vectors:
                        missing_media.setdefault(h, (to_abs(self.root, p.info.path), m))
                continue
            for h, c in zip(p.hashes, p.chunks, strict=True):
                if h not in vectors:
                    missing.setdefault(h, c.text)
        stats.chunks_reused += sum(1 for h in all_hashes if h in vectors)
        transient: set[str] = set()
        if missing:
            done = self._embed_texts(list(missing.items()), transient)
            vectors.update(done)
            stats.chunks_embedded += len(done)
        if missing_media:
            done, media_transient = self._embed_media(missing_media)
            transient |= media_transient
            vectors.update(done)
            stats.chunks_embedded += len(done)

        removed: list[int] = []
        added: list[tuple[int, str, str, np.ndarray]] = []
        now = time.time()
        with self.db_lock:
            with self.store.transaction():
                self._write(pending, vectors, transient, now, removed, added, stats)
            # after COMMIT, still under db_lock: a failed commit leaves the cache untouched,
            # and searches never see the DB and the cache disagree
            self.cache.remove(removed)
            self.cache.add(added)

    def _embed_texts(self, items: list[tuple[str, str]],
                     transient: set[str]) -> dict[str, np.ndarray]:
        """One chunk the model rejects must not abort the update: retry one by one."""
        try:
            vecs = self.embedder.encode_documents([text for _, text in items])
            return dict(zip([h for h, _ in items], vecs, strict=True))
        except Exception as exc:
            if len(items) > 1:
                log.warning("embedding %d chunks failed (%s); retrying one by one",
                            len(items), exc)
                out: dict[str, np.ndarray] = {}
                for one in items:
                    out.update(self._embed_texts([one], transient))
                return out
            log.warning("cannot embed a chunk (%d chars): %s", len(items[0][1]), exc)
            if _is_transient(exc):
                transient.add(items[0][0])
            return {}

    def _write(self, pending: list[_Pending], vectors: dict[str, np.ndarray],
               transient: set[str], now: float, removed: list[int],
               added: list[tuple[int, str, str, np.ndarray]], stats: UpdateStats) -> None:
        for p in pending:
            # a transient failure (GPU out of memory): store a stale mtime and hash so the
            # next scan prepares the file again; finished chunks are reused by hash
            retry = bool(transient.intersection(p.hashes))
            fid = self.store.upsert_file(
                p.info.path, p.info.size, RETRY_MTIME if retry else p.info.mtime_ns,
                RETRY_HASH if retry else p.content_hash, p.info.modality, p.info.lang, now)
            removed.extend(self.store.delete_chunks_of_file(fid))
            rows = self._rows(p, vectors)
            ids = self.store.insert_chunks(fid, rows)
            added.extend((cid, p.info.path, p.info.modality, r.embedding)
                         for cid, r in zip(ids, rows, strict=True))
            if p.content_hash != UNREADABLE:
                stats.files_indexed += 1

    @staticmethod
    def _rows(p: _Pending, vectors: dict[str, np.ndarray]) -> list[NewChunk]:
        path = p.info.path
        if not p.media:
            return [NewChunk(h, p.info.modality, c.start_line, c.end_line, c.symbol, c.kind,
                             c.preview, c.tokens, vectors[h],
                             keywords.document(f"{path} {c.symbol}"),
                             keywords.document(chunk_body(c.text)))
                    for h, c in zip(p.hashes, p.chunks, strict=True)
                    if h in vectors]           # chunks the model rejected are left out
        return [NewChunk(h, m.kind, None, None, m.symbol, MEDIA_KIND_LABEL[m.kind], m.preview, 0,
                         vectors[h], keywords.document(f"{path} {m.symbol}"),
                         keywords.document(m.text), m.page, m.t_start, m.t_end)
                for h, m in zip(p.hashes, p.media, strict=True)
                if h in vectors]               # undecodable chunks are left out
