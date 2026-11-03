"""`semsearch doctor` and `semsearch download-model`: check what semsearch needs and say how to
fix what is missing, and fetch the embedding model only after the user agreed.

Every check is split into gathering facts (imports, files, network) and a pure `judge_*`
function, so the judgements are tested without a GPU, the network or the model.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from semsearch.client import DaemonClient
from semsearch.config import Settings
from semsearch.install import AGENT_FILE, HOOK_MARK, MCP_NAME, MD_START, Targets, script_path

OK, WARN, FAIL = "ok", "warn", "fail"
MIN_PYTHON = (3, 12)
CUDA13_MIN_DRIVER = 580          # PyTorch CUDA 13.0 wheels (pyproject.toml: pytorch-cu130)
MODEL_SIZE_GUESS = 2_000_000_000  # bytes, when the hub cannot tell the download size
NET_TIMEOUT_S = 10.0
NET_ATTEMPTS = 3
WEIGHT_FILES = ("*.safetensors", "pytorch_model*.bin", "*.onnx")
NVIDIA_SMI_TIMEOUT_S = 10.0
FIX_INSTALL_CLAUDE = "run `semsearch install-claude`, then open a new Claude Code session"
FIX_MIRROR = ("check the network or proxy; where huggingface.co is blocked, point HF_ENDPOINT at "
              "a Hugging Face mirror (e.g. https://hf-mirror.com)")


@dataclass(frozen=True)
class Check:
    name: str
    status: str          # OK | WARN | FAIL
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class Context:
    settings: Settings | None    # None: the SEMSEARCH_* variables are invalid
    online: bool


def _gb(size: int | None) -> str:
    return f"{size / 1e9:.1f} GB" if size else "size unknown"


# --- Python / uv ----------------------------------------------------------------------------
def judge_python(version: tuple[int, ...]) -> Check:
    shown = ".".join(map(str, version[:3]))
    if tuple(version[:2]) < MIN_PYTHON:
        return Check("Python", FAIL, f"Python {shown}",
                     "semsearch needs Python 3.12: reinstall with `uv tool install ... "
                     "--python 3.12` or the install script")
    return Check("Python", OK, f"Python {shown}")


def judge_uv(path: str | None) -> Check:
    if path is None:
        return Check("uv", WARN, "uv is not on PATH (only needed to update or uninstall)",
                     "install uv from https://docs.astral.sh/uv/")
    return Check("uv", OK, path)


# --- GPU / PyTorch --------------------------------------------------------------------------
def driver_major(version: str) -> int | None:
    m = re.match(r"\s*(\d+)\.", version)
    return int(m.group(1)) if m else None


def judge_gpu(cuda_build: str | None, cuda_ok: bool, device: str | None,
              driver: str | None) -> Check:
    """cuda_build: torch.version.cuda (None = CPU-only PyTorch); driver: NVIDIA driver version
    from nvidia-smi (None = no NVIDIA GPU found)."""
    if cuda_ok:
        return Check("GPU", OK, f"{device} (CUDA {cuda_build}, driver {driver or '?'})")
    if driver is None:
        return Check("GPU", WARN, "no NVIDIA GPU found: runs on the CPU (works, but indexing "
                                  "is much slower)")
    if cuda_build is None:
        return Check("GPU", WARN, f"NVIDIA GPU found (driver {driver}), but the installed "
                                  "PyTorch is CPU-only, so it is not used",
                     "reinstall semsearch with the install script (it installs the CUDA build)")
    major = driver_major(driver)
    if cuda_build.startswith("13.") and major is not None and major < CUDA13_MIN_DRIVER:
        return Check("GPU", WARN, f"NVIDIA driver {driver} is too old for PyTorch CUDA "
                                  f"{cuda_build}: running on the CPU",
                     f"update the NVIDIA driver to {CUDA13_MIN_DRIVER} or newer, then run "
                     "`semsearch daemon stop`")
    return Check("GPU", WARN, f"NVIDIA driver {driver} found, but PyTorch (CUDA {cuda_build}) "
                              "cannot use the GPU: running on the CPU",
                 "update the NVIDIA driver, or reinstall semsearch with the install script")


def nvidia_driver() -> str | None:
    """Driver version, or None without an NVIDIA GPU. On Windows only System32 is searched:
    a PATH lookup there also tries the current, possibly untrusted, directory."""
    if sys.platform == "win32":
        exe: str | None = str(Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
                              / "System32" / "nvidia-smi.exe")
        exe = exe if Path(exe).is_file() else None
    else:
        exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.strip().splitlines()
    return first[0].strip() if first else None


def check_gpu(_ctx: Context) -> Check:
    import torch
    cuda_ok = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if cuda_ok else None
    return judge_gpu(torch.version.cuda, cuda_ok, device, nvidia_driver())


# --- model ----------------------------------------------------------------------------------
def judge_model(model: str, cached: bool, remote: str, size: int | None) -> Check:
    """remote: ok | skipped (offline) | unreachable | gated | missing."""
    if cached:
        return Check("Model", OK, f"{model} is downloaded")
    if remote in ("ok", "skipped"):
        return Check("Model", WARN, f"{model} is not downloaded yet ({_gb(size)})",
                     "run `semsearch download-model` (otherwise the first search downloads it)")
    if remote == "gated":
        return Check("Model", FAIL, f"{model} needs a Hugging Face account that accepted its "
                                    "license",
                     f"accept the license on https://huggingface.co/{model}, then log in with "
                     "`hf auth login`")
    if remote == "missing":
        return Check("Model", FAIL, f"{model} (or its revision) does not exist on Hugging Face",
                     "check SEMSEARCH_MODEL and SEMSEARCH_MODEL_REVISION; for a private model, "
                     "log in with `hf auth login`")
    return Check("Model", FAIL, f"{model} is not downloaded and Hugging Face cannot be reached",
                 FIX_MIRROR + ", then run `semsearch download-model`")


def model_cached(settings: Settings) -> bool:
    """The snapshot folder alone is not enough: after an interrupted download it can hold the
    small files without the weights."""
    from huggingface_hub import snapshot_download
    try:
        folder = Path(snapshot_download(settings.model, revision=settings.revision,
                                        local_files_only=True))
    except Exception:   # not in the cache, whatever the hub library calls it
        return False
    return (folder / "config.json").is_file() and any(
        any(folder.glob(pattern)) for pattern in WEIGHT_FILES)


def remote_model(settings: Settings) -> tuple[str, int | None]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    api = HfApi()
    for attempt in range(NET_ATTEMPTS):
        try:
            info = api.model_info(settings.model, revision=settings.revision,
                                  files_metadata=True, timeout=NET_TIMEOUT_S)
            api.auth_check(settings.model)
        except GatedRepoError:                    # a subclass of RepositoryNotFoundError
            return "gated", None
        except (RepositoryNotFoundError, RevisionNotFoundError):
            return "missing", None
        except Exception:     # network, proxy, offline mode; a dropped TLS handshake is common
            if attempt == NET_ATTEMPTS - 1:
                return "unreachable", None
            continue
        return "ok", sum(s.size or 0 for s in info.siblings or []) or None
    return "unreachable", None


def fetch_model(settings: Settings) -> str:
    from huggingface_hub import snapshot_download
    return str(snapshot_download(settings.model, revision=settings.revision))


def hub_cache() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def check_model(ctx: Context) -> Check:
    if ctx.settings is None:
        return Check("Model", WARN, "skipped: settings are invalid")
    if model_cached(ctx.settings):
        return judge_model(ctx.settings.model, True, "skipped", None)
    remote, size = remote_model(ctx.settings) if ctx.online else ("skipped", None)
    return judge_model(ctx.settings.model, False, remote, size)


# --- disk / daemon --------------------------------------------------------------------------
def free_bytes(path: Path) -> int:
    for p in (path, *path.parents):     # the folder may not exist yet
        if p.exists():
            return shutil.disk_usage(p).free
    return 0


def judge_disk(free: int, need: int) -> Check:
    if free < need:
        return Check("Disk", FAIL, f"{_gb(free)} free, the model needs {_gb(need)}",
                     "free up space, or move the model cache by setting HF_HOME")
    return Check("Disk", OK, f"{_gb(free)} free")


def check_disk(ctx: Context) -> Check:
    need = 0 if ctx.settings is None or model_cached(ctx.settings) else MODEL_SIZE_GUESS
    return judge_disk(free_bytes(hub_cache()), need)


def judge_daemon(info: dict[str, Any] | None) -> Check:
    if info is None:
        return Check("Daemon", OK, "not running (it starts on the first search)")
    return Check("Daemon", OK, f"running: pid={info.get('pid')} port={info.get('port')}")


def check_daemon(ctx: Context) -> Check:
    if ctx.settings is None:
        return Check("Daemon", WARN, "skipped: settings are invalid")
    return judge_daemon(DaemonClient(ctx.settings, autostart=False).running())


# --- Claude Code registration ---------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    """{} when the file does not exist; ValueError when it exists but cannot be used."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data


def _has_hook(settings: dict[str, Any]) -> bool:
    entries = (settings.get("hooks") or {}).get("PostToolUse") or []
    return any(HOOK_MARK in str(h.get("command", ""))
               for e in entries if isinstance(e, dict)
               for h in e.get("hooks") or [] if isinstance(h, dict))


def judge_claude(t: Targets, expected_mcp: str | None) -> Check:
    try:
        server = (_read_json(t.claude_json).get("mcpServers") or {}).get(MCP_NAME)
        hook = _has_hook(_read_json(t.settings_json))
    except ValueError as exc:
        return Check("Claude Code", WARN, str(exc), "fix the JSON in that file, then "
                     + FIX_INSTALL_CLAUDE)
    try:
        rules = MD_START in t.claude_md.read_text(encoding="utf-8")
    except OSError:
        rules = False
    pieces = {"MCP server": isinstance(server, dict), "hook": hook,
              "subagent": (t.agents_dir / AGENT_FILE).is_file(), "CLAUDE.md rules": rules}
    if not any(pieces.values()):
        return Check("Claude Code", WARN, "semsearch is not registered", FIX_INSTALL_CLAUDE)
    command = str(server.get("command", "")) if isinstance(server, dict) else ""
    # install-claude writes an absolute path; a bare name is resolved by Claude Code via PATH
    if command and Path(command).is_absolute() and not Path(command).is_file():
        return Check("Claude Code", FAIL, f"the registered MCP server {command} does not exist",
                     FIX_INSTALL_CLAUDE)
    missing = [name for name, present in pieces.items() if not present]
    if missing:
        return Check("Claude Code", WARN, "registered, but missing: " + ", ".join(missing),
                     FIX_INSTALL_CLAUDE)
    if expected_mcp and os.path.normcase(command) != os.path.normcase(expected_mcp):
        return Check("Claude Code", WARN, f"registered for another copy of semsearch: {command}",
                     FIX_INSTALL_CLAUDE)
    return Check("Claude Code", OK, "MCP server, hook, subagent and rules are registered")


def check_claude(_ctx: Context) -> Check:
    try:
        expected: str | None = script_path("semsearch-mcp")
    except FileNotFoundError:
        expected = None
    return judge_claude(Targets.default(), expected)


# --- runner / report ------------------------------------------------------------------------
CHECKS: tuple[tuple[str, Callable[[Context], Check]], ...] = (
    ("Python", lambda _ctx: judge_python(tuple(sys.version_info[:3]))),
    ("uv", lambda _ctx: judge_uv(shutil.which("uv"))),
    ("GPU", check_gpu),
    ("Model", check_model),
    ("Disk", check_disk),
    ("Daemon", check_daemon),
    ("Claude Code", check_claude),
)


def run_checks(online: bool) -> list[Check]:
    out: list[Check] = []
    try:
        settings: Settings | None = Settings()
    except ValueError as exc:
        settings = None
        out.append(Check("Settings", FAIL, str(exc),
                         "fix or unset the SEMSEARCH_* environment variable named above"))
    ctx = Context(settings, online)
    for name, fn in CHECKS:
        try:
            out.append(fn(ctx))
        except Exception as exc:   # one broken check must not hide the others
            out.append(Check(name, FAIL, f"check failed: {type(exc).__name__}: {exc}",
                             "please report this at the project's issue tracker"))
    return out


SYMBOLS = {OK: "✓", WARN: "!", FAIL: "✗"}


def format_report(checks: list[Check]) -> str:
    width = max((len(c.name) for c in checks), default=0) + 2
    lines = ["semsearch doctor"]
    for c in checks:
        lines.append(f"  {SYMBOLS[c.status]} {c.name:<{width}}{c.detail}")
        if c.fix:
            lines.append(f"    {'':<{width}}fix: {c.fix}")
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    lines.append("all good" if not (fails or warns)
                 else f"{fails} problem(s), {warns} warning(s)")
    return "\n".join(lines)


def to_json(checks: list[Check]) -> str:
    return json.dumps([asdict(c) for c in checks], ensure_ascii=False, indent=1)


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == FAIL for c in checks) else 0


# --- download-model -------------------------------------------------------------------------
def download_model(settings: Settings, yes: bool, ask: Callable[[str], str]) -> int:
    model = settings.model
    if model_cached(settings):
        print(f"{model} is already downloaded")
        return 0
    remote, size = remote_model(settings)
    if remote != "ok":
        check = judge_model(model, False, remote, size)
        print(f"cannot download: {check.detail}\nfix: {check.fix}")
        return 1
    cache = hub_cache()
    need = size or MODEL_SIZE_GUESS
    if free_bytes(cache) < need:
        print(f"not enough disk space in {cache}: {judge_disk(free_bytes(cache), need).detail}")
        return 1
    if not yes:
        try:
            answer = ask(f"Download {model} ({_gb(size)}) to {cache}? [y/N] ")
        except EOFError:          # no terminal (piped install script): never download unasked
            answer = ""
        if answer.strip().lower() != "y":
            print("not downloaded")
            return 1
    try:
        path = fetch_model(settings)
    except Exception as exc:
        print(f"download failed: {type(exc).__name__}: {exc}\nfix: {FIX_MIRROR}")
        return 1
    print(f"downloaded to {path}")
    return 0
