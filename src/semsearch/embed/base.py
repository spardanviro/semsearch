"""Embedder interface, a deterministic fake for tests, and the factory."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from semsearch.config import EMBED_DIM, Settings

_WORD = re.compile(r"[A-Za-z][a-z0-9]*|[A-Z]+(?![a-z])|\d+|[一-鿿]")


class Embedder(Protocol):
    model_id: str
    load_mode: str

    @property
    def dim(self) -> int: ...     # read-only: a plain attribute or a lazy property both fit

    def count_tokens(self, text: str) -> int: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...


def normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


class FakeEmbedder:
    """Bag-of-words hashing embedder: similar wording -> similar vectors. Tests only."""

    model_id = "fake-hashing"
    load_mode = "text"

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self.documents_encoded = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD.findall(text):
            digest = hashlib.blake2b(word.lower().encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        if not v.any():
            v[0] = 1.0
        return v

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return normalize(np.stack([self._vec(t) for t in texts]))

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.documents_encoded += len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.stack([self._vec(t) for t in texts]))


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder()
    if settings.embedder == "st":
        from semsearch.embed.st_backend import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(settings)
    raise ValueError(f"unknown SEMSEARCH_EMBEDDER {settings.embedder!r} (use 'st' or 'fake')")
