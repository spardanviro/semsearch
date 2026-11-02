"""EmbeddingGemma 2 through sentence-transformers (text-only encoders for now)."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from semsearch.config import EMBED_DIM, MODEL_ID, Settings
from semsearch.embed.base import normalize

log = logging.getLogger(__name__)
QUERY_PROMPT = "SearchQuery"      # PHASE0 + PHASE1_5: beats CodeRetrieval / QuestionAnswering
DOCUMENT_PROMPT = "Document"
PROMPT_OVERHEAD_TOKENS = 12       # "title: none | text: " + special tokens


class SentenceTransformerEmbedder:
    model_id = MODEL_ID
    dim = EMBED_DIM
    load_mode = "text"            # PHASE0: text-only and full loading differ slightly -> never mix

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()       # GPU work
        self._tok_lock = threading.Lock()   # HF fast tokenizers are not safe across threads
        self._model: Any = None
        self._tokenizer: Any = None

    # --- lazy loading ---------------------------------------------------------------------
    def _tok(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        return self._tokenizer

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            t0 = time.perf_counter()
            self._model = SentenceTransformer(
                MODEL_ID, device=device, model_kwargs={"dtype": dtype},
                config_kwargs={"vision_config": None, "audio_config": None})
            self._model.max_seq_length = self.settings.max_tokens
            log.info("loaded %s on %s in %.1fs", MODEL_ID, device, time.perf_counter() - t0)
        return self._model

    # --- api ------------------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        with self._tok_lock:
            return len(self._tok()(text, add_special_tokens=False)["input_ids"])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            model = self._load()
            out = model.encode(texts, prompt_name=QUERY_PROMPT, convert_to_numpy=True,
                               normalize_embeddings=False)
        return normalize(out)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        lengths = [min(self.count_tokens(t), self.settings.max_tokens) + PROMPT_OVERHEAD_TOKENS
                   for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        for batch in self._batches(order, lengths):
            result[batch] = self._encode_batch([texts[i] for i in batch])
        return result

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

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        import torch
        with self._lock:
            model = self._load()
            try:
                out = model.encode(texts, prompt_name=DOCUMENT_PROMPT, batch_size=len(texts),
                                   convert_to_numpy=True, normalize_embeddings=False)
            except torch.OutOfMemoryError:
                if len(texts) == 1:
                    raise
                torch.cuda.empty_cache()
                log.warning("CUDA OOM on batch of %d, splitting", len(texts))
                out = None
        if out is None:
            half = len(texts) // 2
            return np.concatenate([self._encode_batch(texts[:half]),
                                   self._encode_batch(texts[half:])])
        return normalize(out)
