from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from semsearch import cli
from semsearch.client import DaemonUnavailable
from semsearch.config import Settings

SEARCH = {"results": [{"path": "a.py", "lines": [1, 2], "symbol": "f", "kind": "function",
                       "modality": "code", "score": 0.5, "preview": "def f(): pass"}],
          "fresh": True, "index": {"files": 1, "chunks": 1}}


class FakeClient:
    instances: list[FakeClient] = []
    running_info: dict[str, Any] | None = {"pid": 1, "port": 2}
    fail = False

    def __init__(self, *_a: object, **_k: object) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        FakeClient.instances.append(self)

    def _rec(self, name: str, *args: object) -> None:
        self.calls.append((name, args))
        if FakeClient.fail:
            raise DaemonUnavailable("down")

    def search(self, *args: object) -> dict[str, Any]:
        self._rec("search", *args)
        return SEARCH

    def status(self, *args: object) -> dict[str, Any]:
        self._rec("status", *args)
        return {"root": "r", "files": 1, "chunks": 1, "model": "m", "index_path": "p"}

    def reindex(self, *args: object) -> dict[str, Any]:
        self._rec("reindex", *args)
        return {"accepted": [], "full": True}

    def ensure(self) -> dict[str, Any]:
        return {"pid": 1, "port": 2}

    def shutdown(self) -> bool:
        return True

    def running(self) -> dict[str, Any] | None:
        return FakeClient.running_info


@pytest.fixture(autouse=True)
def fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.instances.clear()
    FakeClient.fail = False
    monkeypatch.setattr(cli, "DaemonClient", FakeClient)


def test_cli_search_prints_formatted_results(capsys: pytest.CaptureFixture[str],
                                             tmp_path: Path) -> None:
    code = cli.main(["search", "load", "config", "-k", "3", "-g", "src/**", "--root",
                     str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 0 and "a.py:1-2" in out
    name, args = FakeClient.instances[0].calls[0]
    assert name == "search" and args[1:] == ("load config", 3, ["src/**"], None)


def test_cli_status_json_outputs_raw(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["status", "--json", "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["files"] == 1


def test_cli_reindex_full(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["reindex", "--full", "--root", str(tmp_path)]) == 0
    assert FakeClient.instances[0].calls[0][1][1:] == (True, None)


@pytest.mark.parametrize(("action", "expected"), [("start", "running: pid=1"),
                                                  ("stop", "stopping"),
                                                  ("status", "running: pid=1")])
def test_cli_daemon_actions(action: str, expected: str,
                            capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["daemon", action]) == 0
    assert expected in capsys.readouterr().out


def test_cli_daemon_unavailable_returns_exit_code_2(capsys: pytest.CaptureFixture[str],
                                                   tmp_path: Path) -> None:
    FakeClient.fail = True

    assert cli.main(["status", "--root", str(tmp_path)]) == 2
    assert "error: down" in capsys.readouterr().err


def test_cli_install_dry_run_lists_changes_without_writing(
        capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(cli, "install", lambda *a: called.append("install"))

    assert cli.main(["install-claude", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Planned changes" in out and "semsearch-mcp" in out and called == []


def test_cli_install_declined_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(cli, "install", lambda *a: called.append("install"))
    monkeypatch.setattr("builtins.input", lambda _p: "n")

    assert cli.main(["install-claude"]) == 1
    assert called == []


def test_cli_install_without_a_terminal_counts_as_no(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(cli, "install", lambda *a: called.append("install"))

    def no_tty(_p: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_tty)

    assert cli.main(["install-claude"]) == 1
    assert called == []


def test_cli_uninstall_yes_calls_uninstall(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(cli, "uninstall", lambda *a: called.append("uninstall"))

    assert cli.main(["install-claude", "--uninstall", "--yes"]) == 0
    assert called == ["uninstall"]


def test_settings_env_overrides_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMSEARCH_IDLE_TIMEOUT", "30")
    monkeypatch.setenv("SEMSEARCH_BATCH_TOKENS", "4096")
    s = Settings()
    assert s.idle_timeout_s == 30.0 and s.batch_token_budget == 4096

    monkeypatch.setenv("SEMSEARCH_BATCH_TOKENS", "lots")
    with pytest.raises(ValueError, match="SEMSEARCH_BATCH_TOKENS"):
        Settings()
