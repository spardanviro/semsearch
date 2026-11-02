"""Project root detection, path normalisation and per-project index location."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semsearch.config import Settings


def find_project_root(start: str | os.PathLike[str] | None = None) -> Path:
    """CLAUDE_PROJECT_DIR is used as-is; otherwise walk up from `start` (or cwd) to the
    nearest .git, falling back to the start directory itself."""
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and start is None:
        # never wander up to e.g. a dotfiles repo in the home directory
        return Path(env).resolve()
    base = Path(start or os.getcwd()).resolve()
    home = Path.home().resolve()
    for candidate in (base, *base.parents):
        if candidate == home or candidate in home.parents:
            break     # a dotfiles repo in the home directory is not this project
        if (candidate / ".git").exists():
            return candidate
    return base


def normalize_root(root: str | os.PathLike[str]) -> Path:
    p = Path(root).resolve()
    if not p.is_dir():
        raise ValueError(f"project root is not a directory: {p}")
    return p


def _same_or_inside(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def refuse_root(root: Path, settings: Settings) -> str | None:
    """Reason not to index `root`, or None. The daemon would otherwise index (and serve) any
    directory a client names: the whole home folder, a drive, Claude's own config, or the
    index store itself."""
    home = Path.home().resolve()
    if root == home or root in home.parents:
        return (f"refusing to index {root}: it is (above) the home directory; open Claude "
                "Code in a project folder")
    if root.parent == root:
        return f"refusing to index {root}: a drive or filesystem root"
    for private in (home / ".claude", home / ".ssh", settings.home.resolve()):
        if _same_or_inside(root, private):
            return f"refusing to index {root}: a private configuration directory"
    return None


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
