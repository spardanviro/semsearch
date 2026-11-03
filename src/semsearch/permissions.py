"""Keep the semsearch home (token, indexes with code previews, logs) private to the user.

`os.open(..., 0o600)` means nothing on Windows: files inherit the parent folder's ACL, and
~/.cache is often readable by other local groups (sandbox users, for example). So the home
folder gets an explicit owner-only, inheritable ACL; on POSIX it is chmod 700.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ICACLS_TIMEOUT_S = 60   # /T walks every index file


def _windows_user() -> str:
    user = os.environ.get("USERNAME", "")
    domain = os.environ.get("USERDOMAIN", "")
    return f"{domain}\\{user}" if domain and user else user


def make_private(folder: Path) -> bool:
    """Owner-only access to `folder` and everything inside it. Returns success; never raises
    (a failure is logged: the daemon still works, only less protected)."""
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if os.name != "nt":
            folder.chmod(0o700)
            return True
        user = _windows_user()
        if not user:
            log.warning("cannot restrict %s: USERNAME is not set", folder)
            return False
        # /inheritance:r drops inherited entries; children inherit the new owner-only entry
        if not _icacls(folder, [str(folder), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F"]):
            return False
        # files already inside may carry entries of their own (made by other tools, or before
        # this folder was restricted): reset them to inherit only the owner-only entry
        if any(folder.iterdir()):
            return _icacls(folder, [str(folder / "*"), "/reset", "/T", "/C"])
        return True
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("cannot restrict %s: %s", folder, exc)
        return False


def _icacls(folder: Path, args: list[str]) -> bool:
    proc = subprocess.run(["icacls", *args, "/Q"], capture_output=True, timeout=ICACLS_TIMEOUT_S,
                          stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    if proc.returncode != 0:
        log.warning("cannot restrict %s: icacls exited %s: %s", folder, proc.returncode,
                    proc.stdout.decode("utf-8", "replace")[-300:])
    return proc.returncode == 0
