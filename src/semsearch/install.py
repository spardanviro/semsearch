"""Install / uninstall the Claude Code integration (MCP server, hook, subagent, CLAUDE.md rules)."""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MCP_NAME = "semsearch"
HOOK_MATCHER = "Edit|Write|MultiEdit|NotebookEdit"
HOOK_MARK = "semsearch-hook"
MD_START = "<!-- semsearch:start -->"
MD_END = "<!-- semsearch:end -->"
LEGACY_SECTION = re.compile(r"^# 代码导航\n(?:(?!^# ).*\n?)*", re.M)
INTEGRATIONS = Path(__file__).parent / "integrations"
AGENT_FILE = "semantic-search.md"
MERGE_ATTEMPTS = 3


@dataclass(frozen=True)
class Targets:
    claude_json: Path
    settings_json: Path
    agents_dir: Path
    claude_md: Path

    @classmethod
    def default(cls) -> Targets:
        home = Path.home()
        return cls(home / ".claude.json", home / ".claude" / "settings.json",
                   home / ".claude" / "agents", home / ".claude" / "CLAUDE.md")


def script_path(name: str) -> str:
    """Absolute path of a console script installed next to this interpreter."""
    # no PATH lookup: on Windows that would also search the current (possibly untrusted) directory
    exe = Path(sys.executable).parent / (name + (".exe" if sys.platform == "win32" else ""))
    if exe.exists():
        return str(exe)
    raise FileNotFoundError(f"console script {name!r} not found next to {sys.executable}")


_SHELL_SPECIAL = set('"$`%!\n\r')


def shell_quote(path: str) -> str:
    """Hook commands run through a shell; quote so spaces in the path cannot split it.
    Characters a shell expands even inside double quotes (bash `$(...)`, backticks; cmd
    `%VAR%`, `!`) are refused rather than escaped: such an install path is not worth the risk."""
    bad = sorted(_SHELL_SPECIAL.intersection(path))
    if bad:
        raise ValueError(f"refusing to register a hook from a path containing {bad!r}: {path}")
    return f'"{path}"'


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _backup(path: Path) -> None:
    """`.semsearch.bak` keeps the state before semsearch ever touched the file;
    `.semsearch.prev` keeps the state right before the latest change."""
    if not path.exists():
        return
    original = path.with_name(path.name + ".semsearch.bak")
    if not original.exists():
        shutil.copy2(path, original)
    shutil.copy2(path, path.with_name(path.name + ".semsearch.prev"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    tmp = path.with_name(path.name + ".semsearch.tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        shutil.copymode(path, tmp)   # keep restrictive permissions (e.g. 0600 on POSIX)
    tmp.replace(path)


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


Edit = Callable[[str], str]


def _apply(edits: list[tuple[Path, Edit]]) -> None:
    """Compute every new file first (a bad JSON file aborts before anything is written),
    then write each one; if another program (Claude Code itself writes ~/.claude.json)
    changed a file meanwhile, re-read and re-apply the edit instead of overwriting it."""
    planned = []
    for path, edit in edits:
        stamp = _mtime(path)
        current = path.read_text(encoding="utf-8") if stamp is not None else ""
        planned.append((path, edit, stamp, edit(current)))
    for path, edit, stamp, text in planned:
        for _ in range(MERGE_ATTEMPTS):
            if _mtime(path) == stamp:
                break
            stamp = _mtime(path)
            text = edit(path.read_text(encoding="utf-8") if stamp is not None else "")
        _write(path, text)


def _json_edit(fn: Callable[[dict[str, Any]], dict[str, Any]], path: Path) -> Edit:
    def edit(current: str) -> str:
        data = json.loads(current) if current.strip() else {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        return json.dumps(fn(data), indent=2, ensure_ascii=False) + "\n"
    return edit


# --- individual pieces --------------------------------------------------------------------
def with_mcp(data: dict[str, Any], command: str) -> dict[str, Any]:
    servers = dict(data.get("mcpServers") or {})
    servers[MCP_NAME] = {"type": "stdio", "command": command, "args": [], "env": {}}
    return {**data, "mcpServers": servers}


def without_mcp(data: dict[str, Any]) -> dict[str, Any]:
    servers = {k: v for k, v in (data.get("mcpServers") or {}).items() if k != MCP_NAME}
    return {**data, "mcpServers": servers}


def _is_ours(hook: Any) -> bool:
    return isinstance(hook, dict) and HOOK_MARK in str(hook.get("command", ""))


def _drop_ours(entries: list[Any]) -> list[Any]:
    """Remove only our hook command; an entry the user shares with their own hooks keeps
    them, and only entries left empty by the removal disappear."""
    out = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("hooks"), list):
            out.append(entry)
            continue
        kept = [h for h in entry["hooks"] if not _is_ours(h)]
        if kept or not entry["hooks"]:
            out.append({**entry, "hooks": kept})
    return out


def with_hook(data: dict[str, Any], command: str) -> dict[str, Any]:
    hooks = dict(data.get("hooks") or {})
    post = _drop_ours(hooks.get("PostToolUse") or [])
    post.append({"matcher": HOOK_MATCHER,
                 "hooks": [{"type": "command", "command": command, "timeout": 5}]})
    hooks["PostToolUse"] = post
    return {**data, "hooks": hooks}


def without_hook(data: dict[str, Any]) -> dict[str, Any]:
    hooks = dict(data.get("hooks") or {})
    post = _drop_ours(hooks.get("PostToolUse") or [])
    if post:
        hooks["PostToolUse"] = post
    else:
        hooks.pop("PostToolUse", None)
    return {**data, "hooks": hooks}


def with_md_block(text: str, snippet: str) -> str:
    block = f"{MD_START}\n{snippet.strip()}\n{MD_END}\n"
    pattern = re.compile(re.escape(MD_START) + r".*?" + re.escape(MD_END) + r"\n?", re.S)
    if pattern.search(text):
        return pattern.sub(lambda _m: block, text, count=1)
    if LEGACY_SECTION.search(text):
        return LEGACY_SECTION.sub(lambda _m: block + "\n", text, count=1)
    sep = "" if not text or text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return text + sep + block


def without_md_block(text: str) -> str:
    pattern = re.compile(r"\n?" + re.escape(MD_START) + r".*?" + re.escape(MD_END) + r"\n?", re.S)
    return pattern.sub("\n", text).lstrip("\n")


# --- orchestration ------------------------------------------------------------------------
def describe(t: Targets, mcp_cmd: str, hook_cmd: str) -> list[str]:
    return [f"{t.claude_json}: add MCP server '{MCP_NAME}' -> {mcp_cmd}",
            f"{t.settings_json}: add PostToolUse hook ({HOOK_MATCHER}) -> {hook_cmd}",
            f"{t.agents_dir / 'semantic-search.md'}: write subagent definition",
            f"{t.claude_md}: insert/update the semsearch routing block"]


def install(t: Targets, mcp_cmd: str, hook_cmd: str) -> None:
    agent = (INTEGRATIONS / "semantic-search.md").read_text(encoding="utf-8")
    snippet = (INTEGRATIONS / "claude-md-snippet.md").read_text(encoding="utf-8")
    _apply([
        (t.claude_json, _json_edit(lambda d: with_mcp(d, mcp_cmd), t.claude_json)),
        (t.settings_json, _json_edit(lambda d: with_hook(d, hook_cmd), t.settings_json)),
        (t.agents_dir / AGENT_FILE, lambda _current: agent),   # backed up like the others
        (t.claude_md, lambda current: with_md_block(current, snippet)),
    ])


def uninstall(t: Targets) -> None:
    edits: list[tuple[Path, Edit]] = []
    if t.claude_json.exists():
        edits.append((t.claude_json, _json_edit(without_mcp, t.claude_json)))
    if t.settings_json.exists():
        edits.append((t.settings_json, _json_edit(without_hook, t.settings_json)))
    if t.claude_md.exists():
        edits.append((t.claude_md, without_md_block))
    _apply(edits)
    agent = t.agents_dir / AGENT_FILE
    if agent.exists():
        _backup(agent)
        agent.unlink()
