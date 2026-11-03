"""Request limits for the daemon: body size and request rate.

Only local processes holding the token can reach the API, but one of them going wrong (a
runaway script, a buggy client) must not exhaust the daemon's memory or keep the GPU busy for
everyone else.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from fastapi import HTTPException

MAX_BODY_BYTES = 1_000_000      # the largest legitimate body (1000 notify paths) is ~300 KB
MAX_CONCURRENT_REQUESTS = 32    # uvicorn answers 503 beyond this
RATE_PER_S = 20.0               # sustained requests per second (all token holders together)
RATE_BURST = 100.0              # short bursts allowed on top: a hook firing per edited file

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class BodyLimit:
    """ASGI middleware: reject bodies over `limit` bytes, whether or not Content-Length is
    sent (chunked uploads are counted while they are read)."""

    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1
                if declared < 0 or declared > self.limit:
                    await _reject(send, 413, b'{"detail":"request body too large"}')
                    return
        received = 0

        async def limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    # FastAPI re-raises HTTPException from body parsing: a clean 413
                    raise HTTPException(status_code=413, detail="request body too large")
            return message

        await self.app(scope, limited, send)


async def _reject(send: Send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class RateLimiter:
    """Token bucket shared by all callers. `wait()` -> 0 when the request may go ahead, else
    the seconds until it would be allowed."""

    def __init__(self, rate: float = RATE_PER_S, burst: float = RATE_BURST,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be > 0 and burst >= 1")
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self._tokens = burst
        self._updated = clock()
        self._lock = threading.Lock()

    def wait(self) -> float:
        with self._lock:
            now = self._clock()
            self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate)
            self._updated = now
            if self._tokens >= 1:
                self._tokens -= 1
                return 0.0
            return (1 - self._tokens) / self.rate
