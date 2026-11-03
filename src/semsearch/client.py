"""Client for the daemon: discovers it via daemon.json and starts it when needed."""
from __future__ import annotations

import contextlib
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
MAX_RETRY_WAIT_S = 5.0   # longest pause honoured from a 429/503 Retry-After
STDOUT_LOG = "daemon.stdout.log"
STDOUT_LOG_MAX = 5_000_000


class DaemonUnavailable(RuntimeError):
    pass


class DaemonRequestError(DaemonUnavailable):
    """The daemon is up but rejected the request (bad arguments): not worth a Grep fallback."""


def _http(method: str, url: str, **kwargs: Any) -> httpx.Response:
    # trust_env=False: never route localhost traffic through a system/registry proxy
    with httpx.Client(trust_env=False) as client:
        return client.request(method, url, **kwargs)


def _retry_after(r: httpx.Response) -> float | None:
    """Seconds to wait before retrying a 429/503, or None when the response is final."""
    if r.status_code not in (429, 503):
        return None
    try:
        seconds = float(r.headers.get("retry-after", "1"))
    except ValueError:
        seconds = 1.0
    return min(max(seconds, 0.1), MAX_RETRY_WAIT_S)


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
        """Our daemon answers /health with ok and its pid; a stale daemon.json can point at a
        port some other local service now owns, which must not receive queries or the token."""
        try:
            r = _http("GET", f"http://127.0.0.1:{info['port']}/health", timeout=timeout,
                      headers={"Authorization": f"Bearer {info['token']}"})
            body = r.json() if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return False
        return isinstance(body, dict) and body.get("ok") is True and (
            "pid" not in info or body.get("pid") == info["pid"])

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
            proc = self._start() if owns_lock else None
            deadline = time.monotonic() + START_TIMEOUT_S
            while time.monotonic() < deadline:
                info = self.running()
                if info:
                    return info
                if proc is not None and proc.poll() is not None:
                    # crashed on start (bad SEMSEARCH_* value, broken install): say so now
                    # instead of waiting out the whole timeout
                    raise DaemonUnavailable(f"daemon exited with code {proc.returncode}: "
                                            f"{self._log_tail()}")
                time.sleep(POLL_S)
        finally:
            if owns_lock:   # released on success *and* failure, so a crash never blocks retries
                self.release_lock()
        raise DaemonUnavailable(
            f"daemon did not start within {START_TIMEOUT_S:.0f}s; see {self.settings.log_dir}")

    def _log_tail(self, lines: int = 5) -> str:
        try:
            text = (self.settings.log_dir / STDOUT_LOG).read_text(encoding="utf-8",
                                                                   errors="replace")
        except OSError:
            return f"see {self.settings.log_dir}"
        return " | ".join(text.strip().splitlines()[-lines:]) or f"see {self.settings.log_dir}"

    # --- spawning -------------------------------------------------------------------------
    def _acquire_lock(self) -> bool:
        lock = self.settings.lock_file
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                lock.unlink()
        except FileNotFoundError:
            pass
        except OSError:            # PermissionError: Windows "delete pending" -> treat as taken
            pass
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:  # FileExistsError, or PermissionError while another client deletes it
            return False  # someone else is starting it; just wait for daemon.json
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

    def _start(self) -> subprocess.Popen[Any]:
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.settings.log_dir / STDOUT_LOG
        try:   # only crash output lands here (the daemon logs to daemon.log); keep it small
            if log_path.stat().st_size > STDOUT_LOG_MAX:
                log_path.unlink()
        except OSError:
            pass
        # -I: isolated mode, so the daemon never imports stray .py files from its cwd
        cmd = [sys.executable, "-I", "-m", "semsearch.daemon"]
        with open(log_path, "ab") as out:      # the child keeps its own handle
            kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": out,
                                      "cwd": str(self.settings.home), "close_fds": True}
            if os.name == "nt":
                base = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW)
                try:  # escape the caller's job object so the daemon outlives the Claude session
                    proc = subprocess.Popen(
                        cmd, creationflags=base | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
                except OSError:
                    proc = subprocess.Popen(cmd, creationflags=base, **kwargs)
            else:
                proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
        log.info("started semsearch daemon: %s", cmd)
        return proc

    def release_lock(self) -> None:
        # OSError: Windows sharing violation; the stale-lock rule cleans it up later
        with contextlib.suppress(OSError):
            self.settings.lock_file.unlink(missing_ok=True)

    # --- requests -------------------------------------------------------------------------
    def request(self, path: str, payload: dict[str, Any] | None = None,
                timeout: float = 120.0) -> dict[str, Any]:
        info = self.ensure()
        url = f"http://127.0.0.1:{info['port']}{path}"
        headers = {"Authorization": f"Bearer {info['token']}"}
        deadline = time.monotonic() + timeout
        while True:
            try:
                if payload is None:
                    r = _http("GET", url, headers=headers, timeout=timeout)
                else:
                    r = _http("POST", url, json=payload, headers=headers, timeout=timeout)
            except httpx.HTTPError as exc:
                raise DaemonUnavailable(f"request to daemon failed: {exc}") from exc
            wait = _retry_after(r)
            # rate-limited or at the connection cap: a short wait is better than failing
            if wait is None or time.monotonic() + wait > deadline:
                break
            time.sleep(wait)
        if r.status_code >= 400:
            detail = r.json().get("detail") if r.headers.get("content-type", "").startswith(
                "application/json") else r.text
            if r.status_code in (400, 422):    # bad arguments: the daemon itself is fine
                raise DaemonRequestError(f"invalid request: {detail}")
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
