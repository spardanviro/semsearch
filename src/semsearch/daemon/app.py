"""HTTP API of the semsearch daemon (localhost only, bearer-token protected)."""
from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from semsearch import __version__
from semsearch.config import Settings
from semsearch.embed.base import Embedder
from semsearch.index.project import Project
from semsearch.paths import normalize_root, project_key

log = logging.getLogger(__name__)


@dataclass
class DaemonState:
    settings: Settings
    embedder: Embedder
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.monotonic)
    projects: dict[str, Project] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    opening: dict[str, threading.Lock] = field(default_factory=dict)
    shutdown_requested: threading.Event = field(default_factory=threading.Event)

    def project(self, root: str, create: bool = True) -> Project | None:
        """Open (or reuse) a project. With create=False only projects that already have an
        index are opened, so edits in never-searched projects do not trigger indexing."""
        try:
            path = normalize_root(root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        key = str(path).lower() if os.name == "nt" else str(path)
        with self.lock:
            proj = self.projects.get(key)
            if proj is not None:
                return proj
            key_lock = self.opening.setdefault(key, threading.Lock())
        # opening loads every vector from disk: do it outside the global lock so other
        # projects (and the idle check) are not blocked meanwhile
        with key_lock:
            with self.lock:
                proj = self.projects.get(key)
            if proj is not None:
                return proj
            db = self.settings.projects_dir / project_key(path) / "index.db"
            if not create and not db.exists():
                return None
            log.info("opening project %s", path)
            proj = Project(path, self.settings, self.embedder)
            with self.lock:
                self.projects[key] = proj
            return proj

    def any_busy(self) -> bool:
        with self.lock:
            return any(p.is_busy for p in self.projects.values())

    def close(self) -> None:
        with self.lock:
            for proj in self.projects.values():
                proj.close()
            self.projects.clear()


class RootRequest(BaseModel):
    root: str = Field(min_length=1)


class SearchRequest(RootRequest):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=8, ge=1, le=50)
    path_globs: list[Annotated[str, Field(min_length=1, max_length=200)]] | None = Field(
        default=None, max_length=20)
    modality: str | None = None
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class NotifyRequest(RootRequest):
    paths: list[str] = Field(max_length=1000)


class ReindexRequest(RootRequest):
    full: bool = False
    paths: list[str] | None = Field(default=None, max_length=1000)


LOCAL_HOSTS = ["127.0.0.1", "localhost"]


def create_app(state: DaemonState, allowed_hosts: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="semsearch", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url=None)
    # defence in depth against DNS rebinding (the bearer token already blocks browsers)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS)

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = f"Bearer {state.token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid token")
        state.last_activity = time.monotonic()

    guarded = [Depends(auth)]

    def opened(root: str) -> Project:
        proj = state.project(root)
        if proj is None:  # cannot happen with create=True; keep the type checker honest
            raise HTTPException(status_code=500, detail="project could not be opened")
        return proj

    @app.get("/health", dependencies=guarded)
    def health() -> dict[str, object]:
        return {"ok": True, "pid": os.getpid(), "version": __version__,
                "uptime_s": round(time.time() - state.started_at, 1),
                "projects": len(state.projects)}

    @app.post("/search", dependencies=guarded)
    def search(req: SearchRequest) -> dict[str, object]:
        proj = opened(req.root)
        try:
            results, fresh = proj.search(req.query, req.k, req.path_globs, req.modality,
                                         req.min_score)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"results": [r.to_dict() for r in results], "fresh": fresh,
                "index": proj.status()}

    @app.post("/notify", dependencies=guarded)
    def notify(req: NotifyRequest) -> dict[str, object]:
        proj = state.project(req.root, create=False)
        if proj is None:
            return {"accepted": [], "indexed": False}
        return {"accepted": proj.notify(req.paths), "indexed": True}

    @app.post("/reindex", dependencies=guarded)
    def reindex(req: ReindexRequest) -> dict[str, object]:
        accepted = opened(req.root).reindex(req.full, req.paths)
        return {"accepted": accepted, "full": req.full}

    @app.post("/status", dependencies=guarded)
    def status(req: RootRequest) -> dict[str, object]:
        return opened(req.root).status()

    @app.post("/shutdown", dependencies=guarded)
    def shutdown() -> dict[str, object]:
        state.shutdown_requested.set()
        return {"ok": True}

    return app
