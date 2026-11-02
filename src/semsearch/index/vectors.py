"""In-memory vector matrix for fast brute-force KNN (numpy beats sqlite-vec ~30x, see PHASE0)."""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence

import numpy as np

INITIAL_CAPACITY = 1024
COMPACT_RATIO = 0.3
VectorItem = tuple[int, str, str, np.ndarray]   # (chunk id, path, modality, vector)


class VectorCache:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._lock = threading.RLock()
        self._vecs = np.zeros((INITIAL_CAPACITY, dim), dtype=np.float32)
        self._ids = np.full(INITIAL_CAPACITY, -1, dtype=np.int64)
        self._path_idx = np.zeros(INITIAL_CAPACITY, dtype=np.int32)
        self._modality = np.zeros(INITIAL_CAPACITY, dtype=np.int8)
        self._valid = np.zeros(INITIAL_CAPACITY, dtype=bool)
        self._size = 0
        self._row_of: dict[int, int] = {}
        self._paths: list[str] = []
        self._path_index: dict[str, int] = {}
        self._modalities: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._row_of)

    def _grow(self, need: int) -> None:
        cap = len(self._ids)
        if self._size + need <= cap:
            return
        extra = max(cap * 2, self._size + need) - cap
        self._vecs = np.concatenate([self._vecs, np.zeros((extra, self.dim), dtype=np.float32)])
        self._ids = np.concatenate([self._ids, np.full(extra, -1, dtype=np.int64)])
        self._path_idx = np.concatenate([self._path_idx, np.zeros(extra, dtype=np.int32)])
        self._modality = np.concatenate([self._modality, np.zeros(extra, dtype=np.int8)])
        self._valid = np.concatenate([self._valid, np.zeros(extra, dtype=bool)])

    def _path_id(self, path: str) -> int:
        idx = self._path_index.get(path)
        if idx is None:
            idx = len(self._paths)
            self._paths.append(path)
            self._path_index[path] = idx
        return idx

    def _modality_id(self, modality: str) -> int:
        return self._modalities.setdefault(modality, len(self._modalities))

    def add(self, items: Iterable[VectorItem]) -> None:
        batch = list(items)
        with self._lock:
            self.remove([item[0] for item in batch if item[0] in self._row_of])
            self._grow(len(batch))
            for cid, path, modality, vec in batch:
                row = self._size
                self._vecs[row] = vec
                self._ids[row] = cid
                self._path_idx[row] = self._path_id(path)
                self._modality[row] = self._modality_id(modality)
                self._valid[row] = True
                self._row_of[cid] = row
                self._size += 1

    def remove(self, ids: Iterable[int]) -> None:
        with self._lock:
            for cid in ids:
                row = self._row_of.pop(cid, None)
                if row is not None:
                    self._valid[row] = False
            dead = self._size - len(self._row_of)
            if self._size and dead / self._size > COMPACT_RATIO:
                self._compact()

    def _compact(self) -> None:
        keep = np.flatnonzero(self._valid[:self._size])
        n = len(keep)
        for arr in (self._vecs, self._ids, self._path_idx, self._modality):
            arr[:n] = arr[keep]
        self._valid[:n] = True
        self._valid[n:self._size] = False
        self._size = n
        self._row_of = {int(cid): row for row, cid in enumerate(self._ids[:n])}

    def search(self, query: np.ndarray, k: int,
               path_filter: Callable[[str], bool] | None = None,
               modality: str | None = None) -> list[tuple[int, float]]:
        """Filters are applied before ranking, so a narrow filter never starves the top-k."""
        with self._lock:
            n = self._size
            if n == 0 or k <= 0:
                return []
            mask = self._valid[:n].copy()
            if modality is not None:
                code = self._modalities.get(modality)
                if code is None:
                    return []
                mask &= self._modality[:n] == code
            if path_filter is not None:
                allowed = np.array([path_filter(p) for p in self._paths], dtype=bool)
                mask &= allowed[self._path_idx[:n]]
            scores = self._vecs[:n] @ query.astype(np.float32)
            scores[~mask] = -np.inf
            k = min(k, int(mask.sum()))
            if k == 0:
                return []
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(int(self._ids[r]), float(scores[r])) for r in top]

    def ids(self) -> Sequence[int]:
        with self._lock:
            return list(self._row_of)
