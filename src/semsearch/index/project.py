"""One indexed project: storage, vector cache and a background worker that applies updates."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from semsearch.config import CHUNKER_VERSION, KEYWORDS_VERSION, SCHEMA_VERSION, Settings
from semsearch.embed.base import Embedder
from semsearch.index import scanner
from semsearch.index.indexer import Indexer, UpdateStats, diff
from semsearch.index.search import SearchResult, search
from semsearch.index.store import Store
from semsearch.index.vectors import VectorCache
from semsearch.paths import project_key, to_abs, to_rel

log = logging.getLogger(__name__)
VANISHED_SPARE = 3   # extra results fetched so dropping deleted files still leaves k hits


@dataclass
class _Jobs:
    rebuild: bool = False
    scan: bool = False
    paths: set[str] = field(default_factory=set)

    def empty(self) -> bool:
        return not (self.rebuild or self.scan or self.paths)


class Project:
    def __init__(self, root: Path, settings: Settings, embedder: Embedder) -> None:
        self.root = root
        self.settings = settings
        self.embedder = embedder
        self.key = project_key(root)
        self.db_lock = threading.Lock()
        self.store = Store(settings.projects_dir / self.key / "index.db")
        self.store.ensure_meta(self._expected_meta())
        self.cache = VectorCache(embedder.dim)
        self.cache.add(self.store.iter_vectors())
        self.indexer = Indexer(root, self.store, self.cache, embedder, settings, self.db_lock)

        self._cond = threading.Condition()
        self._jobs = _Jobs()
        self._busy = False
        self._closed = False
        self._last_scan_request = 0.0
        self.last_scan_at: float | None = None
        self.last_stats: UpdateStats | None = None
        self.last_error: str | None = None
        self.progress: tuple[int, int] = (0, 0)
        self._worker = threading.Thread(target=self._run, name=f"indexer-{self.key}", daemon=True)
        self._worker.start()
        self.request_scan(force=True)

    def _expected_meta(self) -> dict[str, str]:
        return {"schema_version": SCHEMA_VERSION, "chunker_version": CHUNKER_VERSION,
                "keywords_version": KEYWORDS_VERSION,
                "model_id": self.embedder.model_id, "dim": str(self.embedder.dim),
                "load_mode": self.embedder.load_mode, "root": str(self.root),
                # document vectors depend on these too (the query prompt does not)
                "document_prompt": ("auto" if self.settings.document_prompt is None
                                    else self.settings.document_prompt),
                "chunk_tokens": f"{self.settings.target_tokens}/{self.settings.max_tokens}"}

    # --- job submission -------------------------------------------------------------------
    def _submit(self, **kinds: object) -> None:
        with self._cond:
            if kinds.get("rebuild"):
                self._jobs.rebuild = True
            if kinds.get("scan"):
                self._jobs.scan = True
            paths = kinds.get("paths")
            if isinstance(paths, (set, list)):
                self._jobs.paths.update(paths)
            self._cond.notify_all()

    def request_scan(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_scan_request < self.settings.scan_cooldown_s:
            return False
        self._last_scan_request = now
        self._submit(scan=True)
        return True

    def notify(self, paths: list[str]) -> list[str]:
        """Queue changed paths (absolute or project-relative). Returns the accepted ones."""
        rels = [r for p in paths if (r := to_rel(self.root, p)) is not None]
        if rels:
            self._submit(paths=set(rels))
        return rels

    def reindex(self, full: bool = False, paths: list[str] | None = None) -> list[str]:
        if full:
            self._submit(rebuild=True)
            return []
        if paths:
            return self.notify(paths)
        self.request_scan(force=True)
        return []

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._busy or not self._jobs.empty():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    @property
    def is_busy(self) -> bool:
        with self._cond:
            return self._busy or not self._jobs.empty()

    # --- worker ---------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cond:
                while self._jobs.empty() and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                jobs, self._jobs = self._jobs, _Jobs()
                self._busy = True
            try:
                self._execute(jobs)
                self.last_error = None
            except Exception as exc:  # keep the worker alive, surface the error in status
                log.exception("indexing %s failed", self.root)
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _execute(self, jobs: _Jobs) -> None:
        if jobs.rebuild:
            with self.db_lock:
                self.store.clear()
            self.cache.remove(self.cache.ids())
        if jobs.rebuild or jobs.scan:
            current = scanner.scan(self.root, self.settings)
            changed, deleted = diff(self.indexer.known_files(), current)
            self.last_scan_at = time.time()
        else:
            known = self.indexer.known_files()
            changed, deleted = [], []
            for rel in sorted(jobs.paths):
                info = scanner.stat_one(self.root, rel, self.settings)
                if info is not None:
                    row = known.get(rel)
                    if row is None or (row.size, row.mtime_ns) != (info.size, info.mtime_ns):
                        changed.append(info)
                elif rel in known:
                    deleted.append(rel)
        if changed or deleted:
            self.progress = (0, len(changed))
            stats = self.indexer.apply(changed, deleted, self._on_progress)
            self.last_stats = stats
            log.info("%s: %s", self.root, asdict(stats))

    def _on_progress(self, done: int, total: int) -> None:
        self.progress = (done, total)

    # --- queries --------------------------------------------------------------------------
    def search(self, query: str, k: int = 8, path_globs: list[str] | None = None,
               modality: str | None = None,
               min_score: float | None = None) -> tuple[list[SearchResult], bool]:
        """-> (results, fresh). Waits briefly for pending updates so results are not stale."""
        self.request_scan()
        fresh = self.wait_idle(self.settings.search_wait_s)
        results = search(self.store, self.cache, self.embedder, self.db_lock, query,
                         k + VANISHED_SPARE, path_globs, modality, min_score,
                         self.settings.per_file_results, self.settings.search_mode,
                         self.settings.keyword_weight)
        return self._drop_vanished(results)[:k], fresh

    def _drop_vanished(self, results: list[SearchResult]) -> list[SearchResult]:
        """Files deleted outside Claude (no hook) inside the scan cooldown must not be returned."""
        gone = {r.path for r in results if not to_abs(self.root, r.path).is_file()}
        if not gone:
            return results
        self.notify(sorted(gone))
        return [r for r in results if r.path not in gone]

    def status(self) -> dict[str, object]:
        with self.db_lock:
            files, chunks = self.store.counts()
        done, total = self.progress
        return {"root": str(self.root), "files": files, "chunks": chunks,
                "indexing": self.is_busy, "progress": {"done": done, "total": total},
                "last_scan_at": self.last_scan_at, "last_error": self.last_error,
                "last_update": asdict(self.last_stats) if self.last_stats else None,
                "model": self.embedder.model_id, "index_path": str(self.store.path)}

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._worker.join(timeout=10)
        self.store.close()
