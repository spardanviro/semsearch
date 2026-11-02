"""Client for the daemon: discovers it via daemon.json and starts it when needed."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from semsearch.config import Settings

log = logging.getLogger(__name__)
START_TIMEOUT_S = 120.0
LOCK_STALE_S = START_TIMEOUT_S + 10   # a starter holds the lock at most START_TIMEOUT_S
POLL_S = 0.25


class DaemonUnavailable(RuntimeError):
    pass


def _http(method: str, url: str, **kwargs: Any) -> httpx.Response:
    # trust_env=False: never route localhost traffic through a system/registry proxy
    with httpx.Client(trust_env=False) as client:
        return client.request(method, url, **kwargs)


class DaemonClient:
    def __init__(self, settings: Settings | None = None, autostart: bool = True) -> None:
        self.settings = settings or Settings()
        self.autostart = autostart

    # --- discovery ------------------------------------------------------------------------
    def _info(self) -> dict[str, Any] | None:
        try:
            info = json.loads(self.settings.daemon_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return info if isinstance(info, dict) and {"port", "token"} <= info.keys() else None

    def _healthy(self, info: dict[str, Any], timeout: float = 2.0) -> bool:
        try:
            r = _http("GET", f"http://127.0.0.1:{info['port']}/health", timeout=timeout,
                      headers={"Authorization": f"Bearer {info['token']}"})
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def running(self) -> dict[str, Any] | None:
        info = self._info()
        return info if info and self._healthy(info) else None

    def ensure(self) -> dict[str, Any]:
        info = self.running()
        if info:
            return info
        if not self.autostart:
            raise DaemonUnavailable("semsearch daemon is not running")
        owns_lock = self._acquire_lock()   # only one client spawns; the others just wait
        try:
            if owns_lock:
                self._start()
            deadline = time.monotonic() + START_TIMEOUT_S
            while time.monotonic() < deadline:
                info = self.running()
                if info:
                    return info
                time.sleep(POLL_S)
        finally:
            if owns_lock:   # released on success *and* failure, so a crash never blocks retries
                self.release_lock()
        raise DaemonUnavailable(
            f"daemon did not start within {START_TIMEOUT_S:.0f}s; see {self.settings.log_dir}")

    # --- spawning -------------------------------------------------------------------------
    def _acquire_lock(self) -> bool:
        lock = self.settings.lock_file
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                lock.unlink()
        except FileNotFoundError:
            pass
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False  # someone else is starting it; just wait for daemon.json
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

    def _start(self) -> None:
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        out = open(self.settings.log_dir / "daemon.stdout.log", "ab")  # noqa: SIM115 (child owns it)
        cmd = [sys.executable, "-m", "semsearch.daemon"]
        kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": out,
                                  "cwd": str(self.settings.home), "close_fds": True}
        if os.name == "nt":
            base = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                    | subprocess.CREATE_NO_WINDOW)
            try:  # escape the caller's job object so the daemon outlives the Claude session
                subprocess.Popen(cmd, creationflags=base | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                                 **kwargs)
            except OSError:
                subprocess.Popen(cmd, creationflags=base, **kwargs)
        else:
            subprocess.Popen(cmd, start_new_session=True, **kwargs)
        out.close()
        log.info("started semsearch daemon: %s", cmd)

    def release_lock(self) -> None:
        self.settings.lock_file.unlink(missing_ok=True)

    # --- requests -------------------------------------------------------------------------
    def request(self, path: str, payload: dict[str, Any] | None = None,
                timeout: float = 120.0) -> dict[str, Any]:
        info = self.ensure()
        url = f"http://127.0.0.1:{info['port']}{path}"
        headers = {"Authorization": f"Bearer {info['token']}"}
        try:
            if payload is None:
                r = _http("GET", url, headers=headers, timeout=timeout)
            else:
                r = _http("POST", url, json=payload, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            raise DaemonUnavailable(f"request to daemon failed: {exc}") from exc
        if r.status_code >= 400:
            detail = r.json().get("detail") if r.headers.get("content-type", "").startswith(
                "application/json") else r.text
            raise DaemonUnavailable(f"daemon error {r.status_code}: {detail}")
        data = r.json()
        if not isinstance(data, dict):
            raise DaemonUnavailable("unexpected daemon response")
        return data

    def search(self, root: Path, query: str, k: int = 8, path_globs: list[str] | None = None,
               modality: str | None = None) -> dict[str, Any]:
        return self.request("/search", {"root": str(root), "query": query, "k": k,
                                        "path_globs": path_globs, "modality": modality})

    def status(self, root: Path) -> dict[str, Any]:
        return self.request("/status", {"root": str(root)})

    def notify(self, root: Path, paths: list[str], timeout: float = 120.0) -> dict[str, Any]:
        return self.request("/notify", {"root": str(root), "paths": paths}, timeout=timeout)

    def reindex(self, root: Path, full: bool = False,
                paths: list[str] | None = None) -> dict[str, Any]:
        return self.request("/reindex", {"root": str(root), "full": full, "paths": paths})

    def shutdown(self) -> bool:
        info = self.running()
        if not info:
            return False
        try:
            _http("POST", f"http://127.0.0.1:{info['port']}/shutdown", timeout=5,
                       headers={"Authorization": f"Bearer {info['token']}"})
        except httpx.HTTPError:
            return False
        return True
