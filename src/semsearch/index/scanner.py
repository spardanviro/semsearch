"""Find the files of a project that should be indexed, with cheap stat info."""
from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pathspec

from semsearch.config import (
    ALWAYS_EXCLUDE_DIRS,
    BUILD_EXCLUDE_DIRS,
    CODE_LANGS,
    CONFIG_EXTS,
    DOC_EXTS,
    MEDIA_EXTS,
    MEDIA_KINDS,
    SENSITIVE_PATTERNS,
    SPECIAL_FILENAMES,
    Settings,
)
from semsearch.globs import safe_spec

log = logging.getLogger(__name__)
LIST_TIMEOUT_S = 120
# the daemon has no console: without this every rg/git call would flash a window on Windows
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def excluded_dirs(root: Path) -> frozenset[str]:
    """Lower-cased directory names to skip at any depth."""
    if (root / ".gitignore").is_file():
        return ALWAYS_EXCLUDE_DIRS
    return ALWAYS_EXCLUDE_DIRS | BUILD_EXCLUDE_DIRS


@dataclass(frozen=True)
class FileInfo:
    path: str          # project-relative posix path
    size: int
    mtime_ns: int
    modality: str      # "code" | "text" | one of MEDIA_KINDS
    lang: str          # language for text; file extension (no dot) for media


def classify(rel_path: str, media: frozenset[str] = frozenset()) -> tuple[str, str] | None:
    """-> (modality, lang) for indexable files, None otherwise. Media files count only when
    their kind is enabled (SEMSEARCH_MEDIA)."""
    name = PurePosixPath(rel_path).name.lower()
    if name in SPECIAL_FILENAMES:
        return "code", SPECIAL_FILENAMES[name]
    ext = PurePosixPath(name).suffix
    if ext in CODE_LANGS:
        return "code", CODE_LANGS[ext]
    if ext in DOC_EXTS:
        return "text", DOC_EXTS[ext]
    if ext in CONFIG_EXTS:
        return "text", CONFIG_EXTS[ext]
    kind = MEDIA_EXTS.get(ext)
    if kind is not None and kind in media:
        return kind, ext.lstrip(".")
    return None


def _size_limit(modality: str, lang: str, settings: Settings) -> int:
    if modality in MEDIA_KINDS:
        return settings.max_media_bytes
    if lang in CONFIG_EXTS.values():
        return settings.max_config_bytes
    return settings.max_file_bytes


class TooManyFiles(RuntimeError):
    pass


# matched against the lower-cased path: Windows file names are case-insensitive, so
# `Credentials.json` and `credentials.json` are the same secret
_SENSITIVE = pathspec.PathSpec.from_lines("gitignore", [p.lower() for p in SENSITIVE_PATTERNS])


def is_sensitive(rel: str) -> bool:
    return _SENSITIVE.match_file(rel.lower())


def _load_ignore(root: Path) -> pathspec.PathSpec:
    custom = root / ".semsearchignore"
    lines = custom.read_text(encoding="utf-8", errors="replace").splitlines() \
        if custom.is_file() else []
    return safe_spec(lines, str(custom))        # repository-controlled: untrusted


def _is_link(st: os.stat_result) -> bool:
    """Symlink, or on Windows any reparse point (junctions included)."""
    reparse = getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(st.st_mode) or bool(reparse)


class _DirGuard:
    """Caches whether a directory really lives inside the root (no symlinked/junction escape)."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._ok: dict[str, bool] = {}

    def inside(self, rel: str) -> bool:
        parent = PurePosixPath(rel).parent.as_posix()
        if parent not in self._ok:
            try:
                real = (self.root / parent).resolve()
                self._ok[parent] = real == self.root or real.is_relative_to(self.root)
            except OSError:
                self._ok[parent] = False
        return self._ok[parent]


def _list_with_rg(root: Path) -> list[str] | None:
    rg = shutil.which("rg")
    if not rg:
        return None
    cmd = [rg, "--files", "--hidden", "--no-require-git", "-g", "!.git"]
    return _run_lister(cmd, root, sep="\n")


def _list_with_git(root: Path) -> list[str] | None:
    git = shutil.which("git")
    if not git or not (root / ".git").exists():
        return None
    cmd = [git, "-c", "core.quotepath=off", "-c", "core.fsmonitor=false",
           "ls-files", "-co", "--exclude-standard", "-z"]
    return _run_lister(cmd, root, sep="\0")


def _run_lister(cmd: list[str], root: Path, sep: str) -> list[str] | None:
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, timeout=LIST_TIMEOUT_S,
                              stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("file lister %s failed: %s", cmd[0], exc)
        return None
    if proc.returncode not in (0, 1):  # rg exits 1 when nothing matched
        log.warning("file lister %s exited %s: %s", cmd[0], proc.returncode,
                    proc.stderr[-300:].decode("utf-8", "replace"))
        return None
    out = proc.stdout.decode("utf-8", "replace")
    return [p.replace("\\", "/") for p in out.split(sep) if p.strip()]


def _list_with_walk(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines() \
        if gitignore.is_file() else []
    spec = safe_spec(lines, str(gitignore))
    skip = excluded_dirs(root)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = [d for d in dirnames
                       if d.lower() not in skip and not spec.match_file(rel_dir + d + "/")
                       and not os.path.isjunction(os.path.join(dirpath, d))
                       and not os.path.islink(os.path.join(dirpath, d))]
        out.extend(rel_dir + f for f in filenames if not spec.match_file(rel_dir + f))
    return out


def list_candidates(root: Path) -> list[str]:
    # git first in a repository: is_gitignored() asks git too, so full scans and hook updates
    # agree (rg drops tracked-but-ignored files that git still lists, and they would flap)
    for lister in (_list_with_git, _list_with_rg):
        files = lister(root)
        if files is not None:
            return files
    return _list_with_walk(root)


def _in_excluded_dir(rel: str, skip: frozenset[str]) -> bool:
    return any(part.lower() in skip for part in PurePosixPath(rel).parts[:-1])


def filter_and_stat(root: Path, rel_paths: Iterable[str], settings: Settings) -> Iterator[FileInfo]:
    ignore = _load_ignore(root)
    guard = _DirGuard(root)
    skip = excluded_dirs(root)
    for rel in rel_paths:
        kind = classify(rel, settings.media)
        # sensitive patterns are checked separately so a repo's ignore file cannot re-include them
        if (kind is None or _in_excluded_dir(rel, skip) or is_sensitive(rel)
                or ignore.match_file(rel)):
            continue
        try:
            st = os.lstat(root / rel)
        except OSError:
            continue  # vanished between listing and stat
        if _is_link(st) or not stat.S_ISREG(st.st_mode) or not guard.inside(rel):
            continue  # never follow links out of the project
        if st.st_size == 0 or st.st_size > _size_limit(kind[0], kind[1], settings):
            continue
        yield FileInfo(rel, st.st_size, st.st_mtime_ns, kind[0], kind[1])


def scan(root: Path, settings: Settings) -> dict[str, FileInfo]:
    files = {f.path: f for f in filter_and_stat(root, list_candidates(root), settings)}
    if len(files) > settings.max_files:
        raise TooManyFiles(
            f"{len(files)} indexable files exceed the limit of {settings.max_files}; add rules to "
            f".semsearchignore or raise SEMSEARCH_MAX_FILES")
    return files


def is_gitignored(root: Path, rel: str) -> bool:
    """Same answer the full scan would give, so hook updates and scans never fight."""
    git = shutil.which("git")
    if git and (root / ".git").exists():
        try:
            proc = subprocess.run([git, "-c", "core.fsmonitor=false", "check-ignore", "-q", "--",
                                   rel],
                                  cwd=root, capture_output=True, timeout=10,
                                  stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
            return proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("git check-ignore failed for %s: %s", rel, exc)
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return False
    spec = safe_spec(gitignore.read_text(encoding="utf-8", errors="replace").splitlines(),
                     str(gitignore))
    return spec.match_file(rel)


def stat_one(root: Path, rel: str, settings: Settings) -> FileInfo | None:
    """Re-check a single path (used for hook notifications)."""
    info = next(iter(filter_and_stat(root, [rel], settings)), None)
    if info is None or is_gitignored(root, rel):
        return None
    return info


def read_text(path: Path, limit: int | None = None) -> str | None:
    """Decode a source file; None when it looks binary or grew beyond `limit` bytes."""
    with open(path, "rb") as fh:
        data = fh.read() if limit is None else fh.read(limit + 1)
    if limit is not None and len(data) > limit:
        return None
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):   # UTF-16 (PowerShell 5 Out-File, SSMS exports)
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    if b"\0" in data[:8192]:
        return None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")
