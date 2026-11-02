"""Embedding models loaded through sentence-transformers (default: EmbeddingGemma 2, text only).

The model, vector size, prompts and token limits come from Settings (SEMSEARCH_MODEL, ...), so
any sentence-transformers model can be used without code changes.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from semsearch.config import EMBED_DIM, MODEL_ID, Settings
from semsearch.embed.base import normalize

log = logging.getLogger(__name__)
PROMPT_SPECIAL_TOKENS = 4         # BOS/EOS etc. added around prompt + text
PROMPT_MARGIN_TOKENS = 32         # sequence-length headroom for prompt + special tokens
# EmbeddingGemma prompt names (PHASE0 + PHASE1_5: SearchQuery beats CodeRetrieval / QA)
GEMMA_PROMPTS = {"query": "SearchQuery", "document": "Document"}


def is_embeddinggemma(model: str) -> bool:
    return "embeddinggemma" in model.lower()


# EmbeddingGemma encoders to skip per load mode (model card: "Selective Encoder Loading").
# PHASE0: vectors differ slightly between load modes, so the mode is part of the index meta.
GEMMA_CONFIG_KWARGS: dict[str, dict[str, None]] = {
    "text": {"vision_config": None, "audio_config": None},
    "text+vision": {"audio_config": None},
    "text+audio": {"vision_config": None},
    "full": {},
}
VISION_KINDS = frozenset({"image", "pdf", "video"})


def load_mode_for(model: str, media: frozenset[str]) -> str:
    """Only the encoders the enabled media kinds need are loaded."""
    if not is_embeddinggemma(model):
        return "default"
    vision, audio = bool(media & VISION_KINDS), "audio" in media
    return {(False, False): "text", (True, False): "text+vision",
            (False, True): "text+audio", (True, True): "full"}[(vision, audio)]


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.model
        self.load_mode = load_mode_for(settings.model, settings.media)
        self._lock = threading.Lock()       # GPU work
        self._tok_lock = threading.Lock()   # HF fast tokenizers are not safe across threads
        self._model: Any = None
        self._tokenizer: Any = None
        self._prompts: tuple[str, str] | None = None   # (query, document) prefix text
        self._dim: int | None = settings.embed_dim or (EMBED_DIM if settings.model == MODEL_ID
                                                       else None)

    # --- lazy loading ---------------------------------------------------------------------
    def _tok(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=self.settings.revision,
                trust_remote_code=self.settings.trust_remote_code)
        return self._tokenizer

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            extra: dict[str, Any] = {}
            if GEMMA_CONFIG_KWARGS.get(self.load_mode):   # skip encoders no media kind needs
                extra["config_kwargs"] = GEMMA_CONFIG_KWARGS[self.load_mode]
            t0 = time.perf_counter()
            if self.settings.trust_remote_code and not self.settings.revision:
                log.warning("SEMSEARCH_TRUST_REMOTE_CODE without SEMSEARCH_MODEL_REVISION: code "
                            "from the latest commit of %s will run", self.model_id)
            model = SentenceTransformer(
                self.model_id, device=device, model_kwargs={"dtype": dtype},
                revision=self.settings.revision,
                trust_remote_code=self.settings.trust_remote_code, **extra)
            model.max_seq_length = self._seq_length(model)
            self._check_dim(model)
            self._prompts = (self._resolve(model, self.settings.query_prompt, "query"),
                             self._resolve(model, self.settings.document_prompt, "document"))
            self._model = model
            log.info("loaded %s on %s in %.1fs (dim %d)", self.model_id, device,
                     time.perf_counter() - t0, self.dim)
        return self._model

    def _check_dim(self, model: Any) -> None:
        getter = (getattr(model, "get_embedding_dimension", None)          # ST >= 6.2
                  or model.get_sentence_embedding_dimension)
        native = int(getter())
        if self._dim is None:
            self._dim = native
        elif self._dim > native:
            raise ValueError(f"SEMSEARCH_EMBED_DIM={self._dim} exceeds {self.model_id}'s "
                             f"vector size {native}")

    def _seq_length(self, model: Any) -> int:
        """Never exceed the model's own limit: longer inputs can crash its position embeddings."""
        native = getattr(model, "max_seq_length", None)
        if self.settings.media and isinstance(native, int) and native > 0:
            # a PDF page (text + 266 image tokens) or a video segment (~3300 tokens) is far
            # longer than a text chunk; truncating them breaks the image/audio token layout.
            # Text chunks stay within max_tokens anyway (the chunker enforces it).
            return native
        # chunk token counts exclude the document prompt and special tokens: leave room so a
        # chunk right at max_tokens is not cut by a few tokens
        want = self.settings.max_tokens + PROMPT_MARGIN_TOKENS
        if isinstance(native, int) and 0 < native < want:
            if native < self.settings.max_tokens:
                log.warning("%s handles at most %d tokens; SEMSEARCH_MAX_TOKENS=%d is capped "
                            "(lower it and SEMSEARCH_CHUNK_TOKENS so chunks are not cut)",
                            self.model_id, native, self.settings.max_tokens)
            return native
        return want

    def _resolve(self, model: Any, prompt: str | None, role: str) -> str:
        """Prompt prefix text for `role` ("query" / "document").

        Unset: EmbeddingGemma's named prompts, otherwise the model card's own `query` /
        `document` prompt if it has one, otherwise none. Set: a prompt name from the model
        card, or literal prefix text."""
        prompts: dict[str, str] = getattr(model, "prompts", None) or {}
        if prompt is None:
            if is_embeddinggemma(self.model_id):
                return prompts.get(GEMMA_PROMPTS[role], "")
            return prompts.get(role, "")
        if prompt in prompts:
            return prompts[prompt]
        if prompts and prompt.isidentifier():
            log.warning("%r is not a prompt name of %s (has: %s); using it as literal text",
                        prompt, self.model_id, ", ".join(sorted(prompts)))
        return prompt

    def _prompt(self, index: int) -> str:
        if self._prompts is None:
            raise RuntimeError("embedding model not loaded")
        return self._prompts[index]

    # --- api ------------------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """Vector size. Unknown sizes need the model, so this may load it (other models only)."""
        if self._dim is None:
            with self._lock:
                self._load()
        if self._dim is None:
            raise RuntimeError(f"could not determine the vector size of {self.model_id}")
        return self._dim

    def count_tokens(self, text: str) -> int:
        with self._tok_lock:
            return len(self._tok()(text, add_special_tokens=False)["input_ids"])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            model = self._load()
            out = model.encode(texts, prompt=self._prompt(0), convert_to_numpy=True,
                               normalize_embeddings=False)
        return self._finish(out)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            self._load()
        overhead = self.count_tokens(self._prompt(1)) + PROMPT_SPECIAL_TOKENS
        lengths = [min(self.count_tokens(t), self.settings.max_tokens) + overhead for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        for batch in self._batches(order, lengths):
            result[batch] = self._encode_batch([texts[i] for i in batch])
        return result

    def _finish(self, out: np.ndarray) -> np.ndarray:
        """float32, truncated to `dim` (Matryoshka-style models), L2-normalized."""
        return normalize(np.asarray(out, dtype=np.float32)[:, :self.dim])

    def _batches(self, order: list[int], lengths: list[int]) -> list[list[int]]:
        """Group indices (sorted by length) so batch_size * longest <= token budget."""
        budget = self.settings.batch_token_budget
        batches: list[list[int]] = []
        current: list[int] = []
        for i in order:
            if current and (len(current) + 1) * lengths[i] > budget:
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)
        return batches

    def encode_media(self, items: list[dict[str, Any]]) -> np.ndarray:
        """Images / PDF pages / audio / video from index.media.load(); no prompt prefix (the
        model card: prefixes apply to text only; PDF pages carry their own `title:` text)."""
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.load_mode in ("text", "default"):
            raise RuntimeError("media embedding needs SEMSEARCH_MEDIA and an EmbeddingGemma model")
        return self._encode_batch(items, media=True)

    def _encode_batch(self, inputs: list[Any], media: bool = False) -> np.ndarray:
        import torch
        with self._lock:
            model = self._load()
            prompt = {} if media else {"prompt": self._prompt(1)}
            try:
                out = model.encode(inputs, batch_size=len(inputs), convert_to_numpy=True,
                                   normalize_embeddings=False, **prompt)
            except torch.OutOfMemoryError:
                if len(inputs) == 1:
                    raise
                torch.cuda.empty_cache()
                log.warning("CUDA OOM on batch of %d, splitting", len(inputs))
                out = None
        if out is None:
            half = len(inputs) // 2
            return np.concatenate([self._encode_batch(inputs[:half], media),
                                   self._encode_batch(inputs[half:], media)])
        return self._finish(out)
