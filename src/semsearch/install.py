"""Install / uninstall the Claude Code integration (MCP server, hook, subagent, CLAUDE.md rules)."""
from __future__ import annotations

import json
import re
import shutil
import sys
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


def shell_quote(path: str) -> str:
    """Hook commands run through a shell; quote so spaces in the path cannot split it."""
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


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# --- individual pieces --------------------------------------------------------------------
def with_mcp(data: dict[str, Any], command: str) -> dict[str, Any]:
    servers = dict(data.get("mcpServers") or {})
    servers[MCP_NAME] = {"type": "stdio", "command": command, "args": [], "env": {}}
    return {**data, "mcpServers": servers}


def without_mcp(data: dict[str, Any]) -> dict[str, Any]:
    servers = {k: v for k, v in (data.get("mcpServers") or {}).items() if k != MCP_NAME}
    return {**data, "mcpServers": servers}


def _is_ours(entry: Any) -> bool:
    return isinstance(entry, dict) and any(
        HOOK_MARK in str(h.get("command", "")) for h in entry.get("hooks") or [])


def with_hook(data: dict[str, Any], command: str) -> dict[str, Any]:
    hooks = dict(data.get("hooks") or {})
    post = [e for e in hooks.get("PostToolUse") or [] if not _is_ours(e)]
    post.append({"matcher": HOOK_MATCHER,
                 "hooks": [{"type": "command", "command": command, "timeout": 5}]})
    hooks["PostToolUse"] = post
    return {**data, "hooks": hooks}


def without_hook(data: dict[str, Any]) -> dict[str, Any]:
    hooks = dict(data.get("hooks") or {})
    post = [e for e in hooks.get("PostToolUse") or [] if not _is_ours(e)]
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
    _write_json(t.claude_json, with_mcp(_load_json(t.claude_json), mcp_cmd))
    _write_json(t.settings_json, with_hook(_load_json(t.settings_json), hook_cmd))
    t.agents_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(INTEGRATIONS / "semantic-search.md", t.agents_dir / "semantic-search.md")
    snippet = (INTEGRATIONS / "claude-md-snippet.md").read_text(encoding="utf-8")
    current = t.claude_md.read_text(encoding="utf-8") if t.claude_md.exists() else ""
    _write(t.claude_md, with_md_block(current, snippet))


def uninstall(t: Targets) -> None:
    if t.claude_json.exists():
        _write_json(t.claude_json, without_mcp(_load_json(t.claude_json)))
    if t.settings_json.exists():
        _write_json(t.settings_json, without_hook(_load_json(t.settings_json)))
    (t.agents_dir / "semantic-search.md").unlink(missing_ok=True)
    if t.claude_md.exists():
        _write(t.claude_md, without_md_block(t.claude_md.read_text(encoding="utf-8")))
