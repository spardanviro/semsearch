"""Project root detection, path normalisation and per-project index location."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath


def find_project_root(start: str | os.PathLike[str] | None = None) -> Path:
    """CLAUDE_PROJECT_DIR is used as-is; otherwise walk up from `start` (or cwd) to the
    nearest .git, falling back to the start directory itself."""
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and start is None:
        # never wander up to e.g. a dotfiles repo in the home directory
        return Path(env).resolve()
    base = Path(start or os.getcwd()).resolve()
    for candidate in (base, *base.parents):
        if (candidate / ".git").exists():
            return candidate
    return base


def normalize_root(root: str | os.PathLike[str]) -> Path:
    p = Path(root).resolve()
    if not p.is_dir():
        raise ValueError(f"project root is not a directory: {p}")
    return p


def project_key(root: Path) -> str:
    """Stable id for a project; case-insensitive on Windows."""
    text = str(root)
    if os.name == "nt":
        text = text.lower()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def to_rel(root: Path, path: str | os.PathLike[str]) -> str | None:
    """Absolute or relative path -> project-relative posix path, or None if outside root."""
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    try:
        rel = p.resolve().relative_to(root)
    except ValueError:
        return None
    return PurePosixPath(*rel.parts).as_posix() if rel.parts else None


def to_abs(root: Path, rel: str) -> Path:
    return root.joinpath(*PurePosixPath(rel).parts)
