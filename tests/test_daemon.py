from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from semsearch.config import Settings
from semsearch.daemon.app import DaemonState, create_app
from semsearch.embed.base import FakeEmbedder


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "net").mkdir(parents=True)
    (root / "net" / "retry.py").write_text(
        "def fetch_with_retry(url):\n    \"\"\"retry download with exponential backoff\"\"\"\n"
        "    return http_get(url)\n", encoding="utf-8")
    return root


@pytest.fixture
def state(tmp_path: Path) -> Iterator[DaemonState]:
    settings = replace(Settings(home=tmp_path / "home", embedder="fake"),
                       scan_cooldown_s=0.0, search_wait_s=10.0)
    st = DaemonState(settings=settings, embedder=FakeEmbedder())
    yield st
    st.close()


@pytest.fixture
def client(state: DaemonState) -> TestClient:
    c = TestClient(create_app(state), base_url="http://127.0.0.1")
    c.headers["Authorization"] = f"Bearer {state.token}"
    return c


def test_daemon_foreign_host_header_is_rejected(state: DaemonState) -> None:
    rebinding = TestClient(create_app(state), base_url="http://evil.example")

    r = rebinding.get("/health", headers={"Authorization": f"Bearer {state.token}"})

    assert r.status_code == 400


def test_daemon_too_many_globs_is_rejected(client: TestClient, project_root: Path) -> None:
    r = client.post("/search", json={"root": str(project_root), "query": "x",
                                     "path_globs": ["*.py"] * 21})

    assert r.status_code == 422


def test_daemon_request_without_token_is_rejected(state: DaemonState) -> None:
    anon = TestClient(create_app(state), base_url="http://127.0.0.1")

    assert anon.get("/health").status_code == 401
    assert anon.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_daemon_search_returns_results_and_index_info(client: TestClient,
                                                      project_root: Path) -> None:
    r = client.post("/search", json={"root": str(project_root), "query": "retry backoff"})

    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["path"] == "net/retry.py"
    assert body["fresh"] is True and body["index"]["files"] == 1


def test_daemon_notify_unindexed_project_does_not_start_indexing(client: TestClient,
                                                                 state: DaemonState,
                                                                 project_root: Path) -> None:
    r = client.post("/notify", json={"root": str(project_root), "paths": ["net/retry.py"]})

    assert r.json() == {"accepted": [], "indexed": False}
    assert state.projects == {}


def test_daemon_notify_indexed_project_accepts_paths(client: TestClient,
                                                     project_root: Path) -> None:
    client.post("/status", json={"root": str(project_root)})

    r = client.post("/notify", json={"root": str(project_root),
                                     "paths": [str(project_root / "net" / "retry.py")]})

    assert r.json() == {"accepted": ["net/retry.py"], "indexed": True}


def test_daemon_bad_root_and_bad_query_return_400(client: TestClient, tmp_path: Path,
                                                  project_root: Path) -> None:
    missing = client.post("/status", json={"root": str(tmp_path / "nope")})
    blank = client.post("/search", json={"root": str(project_root), "query": "   "})
    too_big_k = client.post("/search", json={"root": str(project_root), "query": "x", "k": 500})

    assert missing.status_code == 400
    assert blank.status_code == 400
    assert too_big_k.status_code == 422


def test_daemon_reindex_and_shutdown(client: TestClient, state: DaemonState,
                                     project_root: Path) -> None:
    full = client.post("/reindex", json={"root": str(project_root), "full": True})
    down = client.post("/shutdown")

    assert full.json() == {"accepted": [], "full": True}
    assert down.json() == {"ok": True} and state.shutdown_requested.is_set()


def test_daemon_same_root_different_spelling_shares_project(client: TestClient,
                                                            state: DaemonState,
                                                            project_root: Path) -> None:
    client.post("/status", json={"root": str(project_root)})
    client.post("/status", json={"root": str(project_root / "net" / "..")})

    assert len(state.projects) == 1
