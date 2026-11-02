"""Daemon entry point: `python -m semsearch.daemon` (normally started by the client)."""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from logging.handlers import RotatingFileHandler

import uvicorn

from semsearch.client import DaemonClient
from semsearch.config import Settings
from semsearch.daemon.app import DaemonState, create_app
from semsearch.embed.base import make_embedder
from semsearch.permissions import make_private

log = logging.getLogger("semsearch.daemon")
IDLE_CHECK_S = 15.0
LOG_BYTES = 2_000_000
REPLACE_ATTEMPTS = 8


def setup_logging(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(settings.log_dir / "daemon.log", maxBytes=LOG_BYTES,
                                  backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def write_daemon_file(settings: Settings, port: int, token: str) -> None:
    settings.home.mkdir(parents=True, exist_ok=True)
    tmp = settings.daemon_file.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    body = json.dumps({"pid": os.getpid(), "port": port, "token": token, "started_at": time.time()})
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # token: owner-only on POSIX
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, settings.daemon_file)
            return
        except PermissionError:   # Windows: a client is reading daemon.json right now
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


def remove_daemon_file(settings: Settings) -> None:
    try:
        info = json.loads(settings.daemon_file.read_text(encoding="utf-8"))
        if info.get("pid") == os.getpid():
            settings.daemon_file.unlink()
    except (OSError, ValueError):
        pass


def watch_idle(state: DaemonState, server: uvicorn.Server) -> None:
    while not server.should_exit:
        state.shutdown_requested.wait(IDLE_CHECK_S)
        idle = time.monotonic() - state.last_activity
        if state.shutdown_requested.is_set() or (
                idle > state.settings.idle_timeout_s and not state.any_busy()):
            log.info("stopping (idle %.0fs, shutdown=%s)", idle, state.shutdown_requested.is_set())
            server.should_exit = True


def warm_up(state: DaemonState) -> None:
    """Load the model in the background so the first search is fast."""
    try:
        state.embedder.encode_queries(["warm up"])
    except Exception:
        log.exception("model warm-up failed")


def main() -> None:
    settings = Settings()
    make_private(settings.home)    # before the token file and indexes are (re)written
    setup_logging(settings)
    existing = DaemonClient(settings, autostart=False).running()
    if existing and existing.get("pid") != os.getpid():
        log.info("another daemon (pid %s) is already running; exiting", existing.get("pid"))
        return
    state = DaemonState(settings=settings, embedder=make_embedder(settings))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(create_app(state), log_config=None, access_log=False,
                            timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    write_daemon_file(settings, port, state.token)
    log.info("semsearch daemon pid=%s port=%s embedder=%s", os.getpid(), port, settings.embedder)
    threading.Thread(target=watch_idle, args=(state, server), daemon=True).start()
    threading.Thread(target=warm_up, args=(state,), daemon=True).start()
    try:
        server.run(sockets=[sock])
    finally:
        state.close()
        remove_daemon_file(settings)
        log.info("daemon stopped")


if __name__ == "__main__":
    main()
