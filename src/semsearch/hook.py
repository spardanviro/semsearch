"""Claude Code PostToolUse hook: tell the daemon which file was just edited.

Never blocks or fails the tool call: errors go to logs/hook.log and the exit code is always 0.
If the daemon is not running nothing happens; the next search's scan picks the change up anyway.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any

from semsearch.client import DaemonClient, DaemonUnavailable
from semsearch.config import Settings
from semsearch.paths import find_project_root

NOTIFY_TIMEOUT_S = 0.5
PATH_KEYS = ("file_path", "notebook_path", "path")


def edited_paths(payload: dict[str, Any]) -> list[str]:
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return []
    return [v for k in PATH_KEYS if isinstance(v := tool_input.get(k), str) and v]


def _logger(settings: Settings) -> logging.Logger:
    log = logging.getLogger("semsearch.hook")
    target = str((settings.log_dir / "hook.log").resolve())
    if any(getattr(h, "baseFilename", None) == target for h in log.handlers):
        return log
    for old in list(log.handlers):
        log.removeHandler(old)
        old.close()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(target, maxBytes=500_000, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


def run(stdin_text: str, settings: Settings | None = None) -> int:
    settings = settings or Settings()
    log = _logger(settings)
    try:
        payload = json.loads(stdin_text or "{}")
        paths = edited_paths(payload) if isinstance(payload, dict) else []
        if not paths:
            return 0
        cwd = payload.get("cwd")
        root = find_project_root(None if os.environ.get("CLAUDE_PROJECT_DIR") or not cwd else cwd)
        DaemonClient(settings, autostart=False).notify(root, paths, timeout=NOTIFY_TIMEOUT_S)
    except DaemonUnavailable as exc:
        log.info("daemon not reachable, skipped: %s", exc)
    except Exception:
        log.exception("hook failed")
    return 0


def main() -> None:
    sys.exit(run(sys.stdin.read()))


if __name__ == "__main__":
    main()
