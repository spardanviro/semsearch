"""End-to-end: the client spawns a real daemon process (fake embedder), searches, and stops it."""
from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from semsearch.client import DaemonClient, DaemonUnavailable
from semsearch.config import Settings
from semsearch.hook import run as run_hook

STOP_TIMEOUT_S = 20


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Settings, Path]]:
    home = tmp_path / "home"
    monkeypatch.setenv("SEMSEARCH_HOME", str(home))
    monkeypatch.setenv("SEMSEARCH_EMBEDDER", "fake")
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "save.py").write_text(
        "def save_inventory(player, path):\n    \"\"\"write the player inventory to disk\"\"\"\n"
        "    path.write_text(player.inventory.dump())\n", encoding="utf-8")
    settings = Settings()
    yield settings, root
    DaemonClient(settings, autostart=False).shutdown()
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while settings.daemon_file.exists() and time.monotonic() < deadline:
        time.sleep(0.2)


def test_e2e_client_autostarts_daemon_and_searches(env: tuple[Settings, Path]) -> None:
    settings, root = env
    client = DaemonClient(settings)

    data = client.search(root, "store the player inventory on disk")

    assert data["results"][0]["path"] == "src/save.py"
    assert settings.daemon_file.exists()
    assert not settings.lock_file.exists()


def test_e2e_hook_notifies_running_daemon(env: tuple[Settings, Path]) -> None:
    settings, root = env
    client = DaemonClient(settings)
    client.search(root, "inventory")
    new_file = root / "src" / "load.py"
    new_file.write_text("def load_inventory(path):\n    return parse(path.read_text())\n",
                        encoding="utf-8")

    code = run_hook(f'{{"tool_name": "Write", "cwd": "{root.as_posix()}", '
                    f'"tool_input": {{"file_path": "{new_file.as_posix()}"}}}}', settings)
    time.sleep(0.5)
    data = client.search(root, "read inventory back from file")

    assert code == 0
    assert any(r["path"] == "src/load.py" for r in data["results"])


def test_e2e_no_autostart_raises_when_daemon_absent(env: tuple[Settings, Path]) -> None:
    settings, root = env

    with pytest.raises(DaemonUnavailable):
        DaemonClient(settings, autostart=False).status(root)
