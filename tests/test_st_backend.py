"""SentenceTransformerEmbedder: batching logic with a stub model, plus an opt-in real-GPU test."""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from semsearch.config import Settings
from semsearch.embed.base import make_embedder
from semsearch.embed.st_backend import DOCUMENT_PROMPT, QUERY_PROMPT, SentenceTransformerEmbedder


class StubTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [0] * len(text.split())}


class StubModel:
    def __init__(self, oom_above: int | None = None) -> None:
        self.calls: list[tuple[int, str]] = []
        self.oom_above = oom_above

    def encode(self, texts: list[str], prompt_name: str, **_kw: Any) -> np.ndarray:
        import torch
        self.calls.append((len(texts), prompt_name))
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise torch.OutOfMemoryError("stub OOM")
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, len(t.split()) % 768] = 3.0
        return out


def make(budget: int, model: StubModel) -> SentenceTransformerEmbedder:
    emb = SentenceTransformerEmbedder(replace(Settings(), batch_token_budget=budget))
    emb._tokenizer = StubTokenizer()
    emb._model = model
    return emb


def test_st_encode_documents_groups_batches_by_token_budget() -> None:
    model = StubModel()
    emb = make(budget=100, model=model)
    texts = ["word " * n for n in (5, 80, 6, 7)]

    out = emb.encode_documents(texts)

    assert out.shape == (4, 768)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)
    assert [n for n, _ in model.calls] == [3, 1]          # 3 short ones together, long one alone
    assert all(p == DOCUMENT_PROMPT for _, p in model.calls)
    assert out[1, 80] == pytest.approx(1.0)              # results keep the input order


def test_st_encode_documents_splits_batch_on_oom() -> None:
    model = StubModel(oom_above=2)
    emb = make(budget=10_000, model=model)

    out = emb.encode_documents(["a b", "c d e", "f", "g h i j"])

    assert out.shape == (4, 768)
    assert [n for n, _ in model.calls] == [4, 2, 2]


def test_st_encode_queries_uses_search_prompt_and_empty_docs_ok() -> None:
    model = StubModel()
    emb = make(budget=100, model=model)

    q = emb.encode_queries(["find retry logic"])
    empty = emb.encode_documents([])

    assert q.shape == (1, 768) and model.calls == [(1, QUERY_PROMPT)]
    assert empty.shape == (0, 768)
    assert emb.count_tokens("a b c") == 3


def test_make_embedder_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown SEMSEARCH_EMBEDDER"):
        make_embedder(replace(Settings(), embedder="nope"))
    assert isinstance(make_embedder(replace(Settings(), embedder="st")),
                      SentenceTransformerEmbedder)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("SEMSEARCH_GPU_TESTS") != "1",
                    reason="set SEMSEARCH_GPU_TESTS=1 to run the real model")
def test_st_real_model_ranks_relevant_code_first() -> None:
    emb = SentenceTransformerEmbedder(Settings())
    docs = emb.encode_documents([
        "def fetch_with_retry(url):\n    for i in range(3):\n        try: return get(url)\n"
        "        except Timeout: sleep(2 ** i)",
        "def render_mesh(mesh, camera):\n    draw(mesh.vertices, camera.matrix)",
    ])
    q = emb.encode_queries(["下载失败后重试"])

    scores = q @ docs.T
    assert scores[0, 0] > scores[0, 1]
    assert docs.dtype == np.float32 and docs.shape == (2, 768)
