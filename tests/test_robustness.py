"""Regression tests from the global audit (docs/AUDIT.md): encodings, daemon start-up races,
DB/cache consistency, failure isolation."""
from __future__ import annotations

import io
import sqlite3
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest

from semsearch import cli, hook, mcp_server
from semsearch.client import DaemonClient, DaemonRequestError, DaemonUnavailable
from semsearch.config import Settings
from semsearch.daemon import __main__ as daemon_main
from semsearch.embed.base import FakeEmbedder
from semsearch.index.project import Project


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    write(r, "net/retry.py", "def fetch_with_retry(url):\n    return download(url)\n")
    write(r, "gfx/render.py", "def render_mesh(mesh):\n    return draw(mesh)\n")
    return r


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(Settings(home=tmp_path / "home", embedder="fake"),
                   scan_cooldown_s=0.0, search_wait_s=10.0)


@pytest.fixture
def project(root: Path, settings: Settings) -> Iterator[Project]:
    p = Project(root, settings, FakeEmbedder())
    assert p.wait_idle(10)
    yield p
    p.close()


# --- encodings -------------------------------------------------------------------------------
def test_hook_main_decodes_stdin_as_utf8_whatever_the_console_code_page(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: sys.stdin used the ANSI code page (GBK here) and mangled non-ASCII paths.
    payload = '{"tool_input": {"file_path": "C:\\\\项目\\\\文件.py"}}'.encode()
    seen: list[str] = []
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload), encoding="gbk"))
    monkeypatch.setattr(hook, "run", lambda text, settings=None: seen.append(text) or 0)

    with pytest.raises(SystemExit):
        hook.main()

    assert "项目" in seen[0] and "文件.py" in seen[0]


def test_cli_output_survives_characters_outside_the_console_code_page(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: a piped stdout used GBK and crashed on the first emoji in a preview.
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="gbk"))
    monkeypatch.setattr(cli, "cmd_status", lambda args: print("preview ✓ 😀") or 0)

    assert cli.main(["status"]) == 0
    sys.stdout.flush()
    assert "😀" in raw.getvalue().decode("utf-8")


# --- daemon start-up -------------------------------------------------------------------------
def test_lock_permission_error_counts_as_taken(settings: Settings,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: Windows raises PermissionError (delete pending) instead of FileExistsError.
    import semsearch.client as client_mod

    def busy(*_a: Any, **_k: Any) -> int:
        raise PermissionError("delete pending")

    monkeypatch.setattr(client_mod.os, "open", busy)
    client = DaemonClient(settings)

    assert client._acquire_lock() is False
    client.release_lock()                         # never raises


def test_write_daemon_file_retries_while_a_reader_holds_it(settings: Settings,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: os.replace failed with WinError 5 while a client was reading daemon.json.
    real = daemon_main.os.replace
    calls = {"n": 0}

    def flaky(src: Any, dst: Any) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(daemon_main.os, "replace", flaky)
    daemon_main.write_daemon_file(settings, 1234, "tok")

    assert calls["n"] == 3 and '"port": 1234' in settings.daemon_file.read_text("utf-8")


def test_remove_daemon_file_only_removes_its_own(settings: Settings) -> None:
    settings.home.mkdir(parents=True)
    settings.daemon_file.write_text('{"pid": -1, "port": 1, "token": "x"}', "utf-8")

    daemon_main.remove_daemon_file(settings)

    assert settings.daemon_file.exists()


def test_daemon_that_crashes_on_start_fails_fast(settings: Settings,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: a child that died at once still made the client wait the full 120 s.
    monkeypatch.setenv("SEMSEARCH_HOME", str(settings.home))
    monkeypatch.setenv("SEMSEARCH_EMBEDDER", "no-such-embedder")
    t0 = time.monotonic()

    with pytest.raises(DaemonUnavailable, match="exited"):
        DaemonClient(settings).ensure()

    assert time.monotonic() - t0 < 60


def test_healthy_rejects_a_foreign_service_on_the_port(settings: Settings,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    import semsearch.client as client_mod

    class Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"status": "fine"}             # some other local server answering 200

    monkeypatch.setattr(client_mod, "_http", lambda *a, **k: Resp())

    assert DaemonClient(settings)._healthy({"port": 1, "token": "t", "pid": 7}) is False


def test_mcp_reports_bad_arguments_instead_of_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stub:
        def search(self, *_a: object) -> dict[str, object]:
            raise DaemonRequestError("invalid request: k must be <= 50")

    monkeypatch.setattr(mcp_server, "_client", Stub())

    out = anyio.run(lambda: mcp_server.semantic_search("q", 99))

    assert "rejected the arguments" in out and "Fall back" not in out


def test_mcp_split_globs_keeps_brace_alternatives_together() -> None:
    assert mcp_server.split_globs("**/*.{ts,tsx}, src/**,,!tests/**") == [
        "**/*.{ts,tsx}", "src/**", "!tests/**"]


# --- index consistency -----------------------------------------------------------------------
def test_failed_commit_leaves_cache_matching_the_database(project: Project, root: Path) -> None:
    # Regression: the cache was updated before COMMIT; a failed commit desynced it.
    before = set(project.cache.ids())
    real = project.store.db

    class FailingCommit:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def execute(self, sql: str, *args: Any) -> Any:
            if sql == "COMMIT":
                raise sqlite3.OperationalError("database or disk is full")
            return self.conn.execute(sql, *args)

        @property
        def in_transaction(self) -> bool:
            return self.conn.in_transaction

    write(root, "net/retry.py", "def fetch_with_retry(url):\n    return download(url, tries=5)\n")
    project.store.db = FailingCommit(real)  # type: ignore[assignment]
    try:
        project.notify(["net/retry.py"])
        project.wait_idle(10)
    finally:
        project.store.db = real

    db_ids = {r[0] for r in real.execute("SELECT id FROM chunks")}
    assert set(project.cache.ids()) == db_ids == before


def test_full_rebuild_keeps_the_index_when_the_scan_fails(project: Project) -> None:
    # Regression: rebuild cleared the index first; a failing scan left it empty.
    project.settings = replace(project.settings, max_files=1)   # scan raises TooManyFiles
    files_before = project.status()["files"]

    project.reindex(full=True)
    project.wait_idle(10)

    assert project.status()["files"] == files_before == 2
    assert "TooManyFiles" in str(project.status()["last_error"])


def test_failed_scan_backs_off_and_path_updates_keep_the_error(project: Project,
                                                                root: Path) -> None:
    project.settings = replace(project.settings, max_files=1)
    project.request_scan(force=True)
    project.wait_idle(10)

    assert project.request_scan() is False          # backing off: no rescan on every search
    project.notify([str(root / "net" / "retry.py")])
    project.wait_idle(10)
    assert project.status()["last_error"] is not None


def test_one_chunk_the_model_rejects_does_not_stop_the_others(root: Path,
                                                              settings: Settings) -> None:
    # Regression: one failing encode_documents batch aborted the whole update.
    class Picky(FakeEmbedder):
        def encode_documents(self, texts: list[str]) -> np.ndarray:
            if any("POISON" in t for t in texts):
                raise ValueError("model rejected the input")
            return super().encode_documents(texts)

    write(root, "bad.py", "def poison():\n    return 'POISON'\n")
    p = Project(root, settings, Picky())
    try:
        assert p.wait_idle(10)
        results, _ = p.search("download url retry", k=5)
        assert "net/retry.py" in {r.path for r in results}
        assert p.status()["last_error"] is None
    finally:
        p.close()
