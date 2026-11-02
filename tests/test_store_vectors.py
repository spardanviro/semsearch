import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from semsearch.embed.base import FakeEmbedder, normalize
from semsearch.index.store import NewChunk, Store
from semsearch.index.vectors import VectorCache

DIM = 8


def vec(*values: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(values)] = values
    return normalize(v[None, :])[0]


def new_chunk(h: str, v: np.ndarray, symbol: str = "f") -> NewChunk:
    return NewChunk(h, "code", 1, 3, symbol, "function_definition", "def f(): ...", 5, v)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "idx" / "index.db")
    yield s
    s.close()


def test_store_ensure_meta_mismatch_wipes_index(store: Store) -> None:
    # Arrange
    meta = {"model_id": "m1", "dim": "8"}
    store.ensure_meta(meta)
    fid = store.upsert_file("a.py", 10, 1, "h", "code", "python", 0.0)
    store.insert_chunks(fid, [new_chunk("c1", vec(1))])

    # Act
    unchanged = store.ensure_meta(meta)
    wiped = store.ensure_meta({**meta, "model_id": "m2"})

    # Assert
    assert unchanged is False and wiped is True
    assert store.counts() == (0, 0)
    assert store.meta()["model_id"] == "m2"


def test_store_upsert_and_delete_file_returns_removed_chunk_ids(store: Store) -> None:
    fid = store.upsert_file("a.py", 10, 1, "h1", "code", "python", 0.0)
    ids = store.insert_chunks(fid, [new_chunk("c1", vec(1)), new_chunk("c2", vec(0, 1))])

    same = store.upsert_file("a.py", 12, 2, "h2", "code", "python", 1.0)
    removed = store.delete_file("a.py")

    assert same == fid
    assert sorted(removed) == sorted(ids)
    assert store.files() == {} and store.counts() == (0, 0)
    assert store.delete_file("missing.py") == []


def test_store_embeddings_by_hash_and_get_chunks_roundtrip(store: Store) -> None:
    fid = store.upsert_file("pkg/a.py", 10, 1, "h", "code", "python", 0.0)
    [cid] = store.insert_chunks(fid, [new_chunk("abc", vec(3, 4), "Foo.bar")])

    by_hash = store.embeddings_by_hash(["abc", "nope", "abc"])
    hits = store.get_chunks([cid])
    vectors = list(store.iter_vectors())

    assert set(by_hash) == {"abc"}
    np.testing.assert_allclose(by_hash["abc"], vec(3, 4))
    assert hits[cid].path == "pkg/a.py" and hits[cid].symbol == "Foo.bar"
    assert vectors[0][:3] == (cid, "pkg/a.py", "code")


def test_store_failed_commit_rolls_back_and_connection_stays_usable(store: Store) -> None:
    # Regression: a COMMIT error left the shared connection inside an open transaction.
    class FailingCommit:
        def __init__(self, real: sqlite3.Connection) -> None:
            self.real = real

        def execute(self, sql: str, *args: Any) -> object:
            if sql == "COMMIT":
                raise sqlite3.OperationalError("disk full")
            return self.real.execute(sql, *args)

        @property
        def in_transaction(self) -> bool:
            return self.real.in_transaction

    real = store.db
    store.db = FailingCommit(real)  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError), store.transaction():
        real.execute("INSERT INTO meta(key, value) VALUES ('x', 'y')")
    store.db = real

    assert not real.in_transaction
    with store.transaction():
        store.upsert_file("a.py", 1, 1, "h", "code", "python", 0.0)
    assert "x" not in store.meta() and "a.py" in store.files()


def test_store_schema_version_change_recreates_tables(store: Store) -> None:
    store.ensure_meta({"schema_version": "1"})
    store.db.execute("CREATE TABLE IF NOT EXISTS leftover(x)")
    fid = store.upsert_file("a.py", 1, 1, "h", "code", "python", 0.0)
    store.insert_chunks(fid, [new_chunk("c", vec(1))])

    wiped = store.ensure_meta({"schema_version": "2"})

    assert wiped and store.counts() == (0, 0) and store.meta() == {"schema_version": "2"}


def test_store_transaction_error_rolls_back(store: Store) -> None:
    with pytest.raises(RuntimeError), store.transaction():
        store.upsert_file("a.py", 1, 1, "h", "code", "python", 0.0)
        raise RuntimeError("boom")

    assert store.files() == {}


def test_vector_cache_search_returns_best_first_and_respects_filter() -> None:
    cache = VectorCache(DIM)
    cache.add([(1, "a.py", "code", vec(1, 0)), (2, "b.py", "code", vec(0, 1)),
               (3, "b.py", "text", vec(1, 1))])

    top = cache.search(vec(1, 0.1), k=2)
    only_b = cache.search(vec(1, 0), k=5, path_filter=lambda p: p == "b.py")
    only_text = cache.search(vec(1, 0), k=1, modality="text")

    assert [cid for cid, _ in top] == [1, 3]
    assert top[0][1] > top[1][1]
    assert [cid for cid, _ in only_b] == [3, 2]
    assert [cid for cid, _ in only_text] == [3]          # filter applied before top-k
    assert cache.search(vec(1, 0), k=5, modality="video") == []


def test_vector_cache_remove_and_compact_keep_results_consistent() -> None:
    cache = VectorCache(DIM)
    cache.add([(i, f"f{i}.py", "code", vec(1, i)) for i in range(10)])

    cache.remove(range(0, 8))           # triggers compaction (>30% dead)
    cache.add([(100, "new.py", "code", vec(1, 0))])

    ids = {cid for cid, _ in cache.search(vec(1, 0), k=10)}
    assert ids == {8, 9, 100}
    assert len(cache) == 3


def test_vector_cache_grows_beyond_initial_capacity() -> None:
    cache = VectorCache(DIM)
    cache.add([(i, "x.py", "code", vec(1, i % 7)) for i in range(3000)])

    assert len(cache) == 3000
    assert len(cache.search(vec(1), k=5)) == 5


def test_vector_cache_readding_id_replaces_vector() -> None:
    cache = VectorCache(DIM)
    cache.add([(1, "a.py", "code", vec(1, 0))])
    cache.add([(1, "a.py", "code", vec(0, 1))])

    [(cid, score)] = cache.search(vec(0, 1), k=5)
    assert cid == 1 and score == pytest.approx(1.0)


def test_vector_cache_empty_or_fully_filtered_returns_nothing() -> None:
    cache = VectorCache(DIM)
    assert cache.search(vec(1), k=3) == []
    cache.add([(1, "a.py", "code", vec(1))])
    assert cache.search(vec(1), k=3, path_filter=lambda _p: False) == []


def test_fake_embedder_similar_text_scores_higher() -> None:
    emb = FakeEmbedder()
    docs = emb.encode_documents(["retry the http request with backoff",
                                 "render the player mesh on screen"])
    [q] = emb.encode_queries(["http retry backoff"])

    assert q @ docs[0] > q @ docs[1]
    assert emb.documents_encoded == 2
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-5)

