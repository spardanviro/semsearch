"""SentenceTransformerEmbedder: batching, prompts and dimensions with a stub model, plus an
opt-in real-GPU test."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from semsearch.config import EMBED_DIM, MODEL_ID, Settings
from semsearch.embed.base import make_embedder
from semsearch.embed.st_backend import SentenceTransformerEmbedder

PROMPTS = {"SearchQuery": "task: search result | query: ", "Document": "title: none | text: "}


class StubTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [0] * len(text.split())}


class StubModel:
    def __init__(self, oom_above: int | None = None, native_dim: int = 768) -> None:
        self.calls: list[tuple[int, str]] = []
        self.oom_above = oom_above
        self.native_dim = native_dim
        self.prompts = dict(PROMPTS)

    def get_sentence_embedding_dimension(self) -> int:
        return self.native_dim

    def encode(self, texts: list[str], prompt: str, **_kw: Any) -> np.ndarray:
        import torch
        self.calls.append((len(texts), prompt))
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise torch.OutOfMemoryError("stub OOM")
        out = np.zeros((len(texts), self.native_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, len(t.split()) % 4] = 3.0     # within the first 4 dims: survives truncation
            out[i, -1] = 4.0                      # beyond them: dropped by truncation
        return out


def make(stub: StubModel, **overrides: Any) -> SentenceTransformerEmbedder:
    emb = SentenceTransformerEmbedder(replace(Settings(), **overrides))
    emb._tokenizer = StubTokenizer()
    emb._check_dim(stub)
    emb._prompts = (emb._resolve(stub, emb.settings.query_prompt, "query"),
                    emb._resolve(stub, emb.settings.document_prompt, "document"))
    emb._model = stub
    return emb


def test_st_encode_documents_groups_batches_by_token_budget() -> None:
    model = StubModel()
    emb = make(model, batch_token_budget=100)
    texts = ["word " * n for n in (5, 80, 6, 7)]

    out = emb.encode_documents(texts)

    assert out.shape == (4, 768)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)
    assert [n for n, _ in model.calls] == [3, 1]          # 3 short ones together, long one alone
    assert all(p == PROMPTS["Document"] for _, p in model.calls)
    assert out[1, 0] > 0                                  # results keep the input order (80 % 4)


def test_st_encode_documents_splits_batch_on_oom() -> None:
    model = StubModel(oom_above=2)
    emb = make(model, batch_token_budget=10_000)

    out = emb.encode_documents(["a b", "c d e", "f", "g h i j"])

    assert out.shape == (4, 768)
    assert [n for n, _ in model.calls] == [4, 2, 2]


def test_st_encode_queries_uses_named_prompt_and_empty_docs_ok() -> None:
    model = StubModel()
    emb = make(model)

    q = emb.encode_queries(["find retry logic"])
    empty = emb.encode_documents([])

    assert q.shape == (1, 768) and model.calls == [(1, PROMPTS["SearchQuery"])]
    assert empty.shape == (0, 768)
    assert emb.count_tokens("a b c") == 3


class SpecialTokenizer(StubTokenizer):
    all_special_tokens = ["<bos>", "<|image|>", "<|image>"]


class MediaTokenModel(StubModel):
    """Like the EmbeddingGemma 2 processor: a literal <|image|> in text means "an image goes
    here" and the call fails when no image is passed."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    def encode(self, texts: list[str], prompt: str, **kw: Any) -> np.ndarray:
        if any("<|image|>" in t for t in texts):
            raise ValueError("Found 1 <|image|> tokens in the text but no images were passed.")
        self.texts.extend(texts)
        return super().encode(texts, prompt, **kw)


def test_st_documents_with_literal_special_tokens_still_get_vectors() -> None:
    # Arrange: source code that mentions the model's own control tokens
    model = MediaTokenModel()
    emb = make(model)
    emb._tokenizer = SpecialTokenizer()
    docs = ['PDF_TEXT = "title: x | text: <|image|>"', "tokens = ['<bos>', '<|image>']"]

    # Act
    out = emb.encode_documents(docs)

    # Assert: every chunk embedded, and the model never saw a raw control token
    assert out.shape == (2, 768)
    assert len(model.texts) == 2
    assert not any(tok in t for t in model.texts for tok in SpecialTokenizer.all_special_tokens)
    assert all("image" in t or "bos" in t for t in model.texts)   # the words themselves stay


def test_st_queries_with_literal_special_tokens_are_neutralized() -> None:
    model = MediaTokenModel()
    emb = make(model)
    emb._tokenizer = SpecialTokenizer()

    out = emb.encode_queries(["where is <|image|> inserted into the PDF text"])

    assert out.shape == (1, 768)
    assert "<|image|>" not in model.texts[0]


def test_st_literal_prompt_text_is_used_when_not_a_prompt_name() -> None:
    model = StubModel()
    emb = make(model, query_prompt="query: ", document_prompt="")

    emb.encode_queries(["x"])
    emb.encode_documents(["y"])

    assert [p for _, p in model.calls] == ["query: ", ""]


def test_st_embed_dim_truncates_and_renormalizes() -> None:
    emb = make(StubModel(), embed_dim=4)

    out = emb.encode_queries(["a b c"])

    assert emb.dim == 4 and out.shape == (1, 4)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    assert out[0, 3] == pytest.approx(1.0)               # the dropped last dim did not leak in


def test_st_dim_comes_from_the_model_for_non_default_models() -> None:
    known = SentenceTransformerEmbedder(Settings())
    other = make(StubModel(native_dim=1024), model="BAAI/bge-m3")

    assert known.dim == EMBED_DIM and known.model_id == MODEL_ID     # no model load needed
    assert other.dim == 1024 and other.model_id == "BAAI/bge-m3"


def test_st_embed_dim_larger_than_model_is_rejected() -> None:
    emb = SentenceTransformerEmbedder(replace(Settings(), model="BAAI/bge-m3", embed_dim=2048))

    with pytest.raises(ValueError, match="SEMSEARCH_EMBED_DIM"):
        emb._check_dim(StubModel())


def test_st_load_mode_is_text_only_for_embeddinggemma() -> None:
    gemma = SentenceTransformerEmbedder(Settings())
    other = SentenceTransformerEmbedder(replace(Settings(), model="BAAI/bge-m3"))

    assert gemma.load_mode == "text" and other.load_mode == "default"


@pytest.mark.parametrize(("env", "value", "message"), [
    ("SEMSEARCH_CHUNK_TOKENS", "2000", "SEMSEARCH_CHUNK_TOKENS"),
    ("SEMSEARCH_EMBED_DIM", "-1", "SEMSEARCH_EMBED_DIM"),
    ("SEMSEARCH_TRUST_REMOTE_CODE", "maybe", "SEMSEARCH_TRUST_REMOTE_CODE"),
    ("SEMSEARCH_MODEL", "  ", "SEMSEARCH_MODEL"),
    ("SEMSEARCH_EMBED_DIM", "1024", "exceeds"),      # default model is 768-d: fail before any wipe
    ("SEMSEARCH_MAX_TOKENS", "16", "SEMSEARCH_MAX_TOKENS"),
])
def test_settings_reject_invalid_model_options(monkeypatch: pytest.MonkeyPatch, env: str,
                                               value: str, message: str) -> None:
    monkeypatch.setenv(env, value)

    with pytest.raises(ValueError, match=message):
        Settings()


def test_settings_read_model_options_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {"SEMSEARCH_MODEL": "Qwen/Qwen3-Embedding-0.6B", "SEMSEARCH_EMBED_DIM": "512",
                        "SEMSEARCH_QUERY_PROMPT": "query", "SEMSEARCH_TRUST_REMOTE_CODE": "yes",
                        "SEMSEARCH_CHUNK_TOKENS": "400", "SEMSEARCH_MAX_TOKENS": "512"}.items():
        monkeypatch.setenv(name, value)

    s = Settings()

    assert (s.model, s.embed_dim, s.query_prompt, s.trust_remote_code) == \
        ("Qwen/Qwen3-Embedding-0.6B", 512, "query", True)
    assert (s.target_tokens, s.max_tokens) == (400, 512)


# --- real _load() path with a stand-in SentenceTransformer -----------------------------------
class FakeSentenceTransformer:
    """Records how _load() constructs the model; behaves like a loaded ST model."""
    prompts: dict[str, str] = {}
    native_dim = 1024
    native_len = 8192
    created: list[FakeSentenceTransformer] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name, self.kwargs = name, kwargs
        self.prompts = dict(type(self).prompts)
        self.max_seq_length = type(self).native_len
        type(self).created.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return type(self).native_dim

    def encode(self, texts: list[str], prompt: str, **_kw: Any) -> np.ndarray:
        return np.ones((len(texts), type(self).native_dim), dtype=np.float32)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    import types
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    monkeypatch.setattr(FakeSentenceTransformer, "created", [])
    return FakeSentenceTransformer


def loaded(**overrides: Any) -> tuple[SentenceTransformerEmbedder, FakeSentenceTransformer]:
    emb = SentenceTransformerEmbedder(replace(Settings(), **overrides))
    emb.encode_queries(["warm up"])
    return emb, FakeSentenceTransformer.created[-1]


def test_load_other_model_uses_its_own_query_document_prompts(
        fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: unset prompts fell back to EmbeddingGemma's names, which other models then
    # received as literal "SearchQuery"/"Document" prefixes.
    monkeypatch.setattr(fake_st, "prompts", {"query": "Instruct: find code\nQuery: ",
                                             "document": ""})
    emb, model = loaded(model="Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

    assert emb._prompts == ("Instruct: find code\nQuery: ", "")
    assert "config_kwargs" not in model.kwargs and model.kwargs["trust_remote_code"] is True
    assert emb.dim == 1024


def test_load_model_without_prompts_uses_none(fake_st: type[FakeSentenceTransformer],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fake_st, "prompts", {})
    emb, _ = loaded(model="BAAI/bge-m3")

    assert emb._prompts == ("", "")


def test_load_unknown_prompt_name_warns_and_is_literal(
        fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(fake_st, "prompts", {"query": "q: "})
    emb, _ = loaded(model="BAAI/bge-m3", query_prompt="SearchQuery")

    assert emb._prompt(0) == "SearchQuery"
    assert "not a prompt name" in caplog.text


def test_load_embeddinggemma_text_only_and_named_prompts(
        fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fake_st, "prompts", PROMPTS)
    monkeypatch.setattr(fake_st, "native_dim", 768)
    emb, model = loaded()

    assert model.kwargs["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert emb._prompts == (PROMPTS["SearchQuery"], PROMPTS["Document"])


def test_load_caps_sequence_length_at_the_model_limit(
        fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    # Regression: max_tokens overwrote a shorter native limit -> position-embedding crashes.
    monkeypatch.setattr(fake_st, "native_len", 512)
    _, model = loaded(model="sentence-transformers/all-MiniLM-L6-v2")

    assert model.max_seq_length == 512
    assert "capped" in caplog.text


@pytest.mark.parametrize(("media", "mode", "skipped"), [
    (frozenset(), "text", {"vision_config": None, "audio_config": None}),
    (frozenset({"image", "pdf"}), "text+vision", {"audio_config": None}),
    (frozenset({"audio"}), "text+audio", {"vision_config": None}),
    (frozenset({"video", "audio"}), "full", None),
])
def test_load_only_encoders_the_media_kinds_need(
        fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch,
        media: frozenset[str], mode: str, skipped: dict[str, None] | None) -> None:
    monkeypatch.setattr(fake_st, "prompts", PROMPTS)
    monkeypatch.setattr(fake_st, "native_dim", 768)
    emb, model = loaded(media=media)

    assert emb.load_mode == mode
    assert model.kwargs.get("config_kwargs") == skipped


def test_encode_media_passes_no_prompt_and_needs_media_mode(
        fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(fake_st, "native_dim", 768)
    monkeypatch.setattr(fake_st, "encode", lambda self, items, **kw: (
        calls.append(kw), np.ones((len(items), 768), dtype=np.float32))[1])
    text_only = SentenceTransformerEmbedder(Settings())
    with_media = SentenceTransformerEmbedder(replace(Settings(), media=frozenset({"image"})))

    out = with_media.encode_media([{"image": "a"}, {"image": "b"}])

    assert out.shape == (2, 768) and "prompt" not in calls[-1]
    with pytest.raises(RuntimeError, match="SEMSEARCH_MEDIA"):
        text_only.encode_media([{"image": "a"}])


def test_dim_of_unknown_model_loads_it_once(fake_st: type[FakeSentenceTransformer]) -> None:
    emb = SentenceTransformerEmbedder(replace(Settings(), model="BAAI/bge-m3"))

    assert emb.dim == 1024 and emb.dim == 1024
    assert len(fake_st.created) == 1


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
