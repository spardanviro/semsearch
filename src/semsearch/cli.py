"""`semsearch` command line: search / status / reindex / daemon control / Claude Code install /
environment check / model download."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semsearch import doctor
from semsearch.client import DaemonClient, DaemonUnavailable
from semsearch.config import MEDIA_KINDS, Settings
from semsearch.formatting import format_search, format_status
from semsearch.install import Targets, describe, install, script_path, shell_quote, uninstall
from semsearch.paths import find_project_root


def _root(args: argparse.Namespace) -> Path:
    return find_project_root(args.root) if args.root else find_project_root()


def cmd_search(args: argparse.Namespace) -> int:
    globs = [g for g in (args.glob or []) if g]
    data = DaemonClient().search(_root(args), " ".join(args.query), args.k, globs or None,
                                 args.modality)
    print(json.dumps(data, indent=1, ensure_ascii=False) if args.json
          else format_search(" ".join(args.query), data))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    data = DaemonClient().status(_root(args))
    print(json.dumps(data, indent=1, ensure_ascii=False) if args.json else format_status(data))
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    # paths are typed relative to the shell's cwd, not to the project root
    paths = [str(Path(p).resolve()) for p in args.paths] if args.paths else None
    data = DaemonClient().reindex(_root(args), args.full, paths)
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
    elif args.full:
        print("full re-index started")
    else:
        accepted = data.get("accepted") or []
        print(f"queued {len(accepted)} of {len(paths)} paths" if paths else "rescan queued")
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    client = DaemonClient()
    if args.action == "start":
        info = client.ensure()
        print(f"running: pid={info.get('pid')} port={info.get('port')}")
    elif args.action == "stop":
        print("stopping" if client.shutdown() else "not running")
    else:
        info = client.running()
        print(f"running: pid={info.get('pid')} port={info.get('port')}" if info else "not running")
    return 0


def _confirm(question: str) -> bool:
    try:
        return input(question).strip().lower() == "y"
    except EOFError:      # no terminal (e.g. run from a piped install script): never assume yes
        print()
        return False


def cmd_install(args: argparse.Namespace) -> int:
    targets = Targets.default()
    if args.uninstall:
        print("Will remove the semsearch MCP server, hook, subagent and CLAUDE.md block.")
        if not args.yes and not _confirm("Proceed? [y/N] "):
            return 1
        uninstall(targets)
        print("removed (backups: *.semsearch.bak)")
        return 0
    mcp_cmd = script_path("semsearch-mcp")                  # executed directly, no shell
    hook_cmd = shell_quote(script_path("semsearch-hook"))   # executed through a shell
    print("Planned changes:")
    for line in describe(targets, mcp_cmd, hook_cmd):
        print("  - " + line)
    if args.dry_run:
        return 0
    if not args.yes and not _confirm("Proceed? [y/N] "):
        return 1
    install(targets, mcp_cmd, hook_cmd)
    print("installed (backups: *.semsearch.bak). Restart Claude Code sessions to load it.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = doctor.run_checks(online=not args.offline)
    print(doctor.to_json(checks) if args.json else doctor.format_report(checks))
    return doctor.exit_code(checks)


def cmd_download_model(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return doctor.download_model(settings, yes=args.yes, ask=input)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semsearch", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="semantic search in a project")
    s.add_argument("query", nargs="+")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("-g", "--glob", action="append", help="path glob filter, repeatable")
    s.add_argument("-m", "--modality", choices=["code", "text", *MEDIA_KINDS])
    s.set_defaults(func=cmd_search)

    st = sub.add_parser("status", help="index status")
    st.set_defaults(func=cmd_status)

    r = sub.add_parser("reindex", help="rescan, re-index paths, or rebuild everything")
    r.add_argument("paths", nargs="*")
    r.add_argument("--full", action="store_true")
    r.set_defaults(func=cmd_reindex)

    d = sub.add_parser("daemon", help="control the background daemon")
    d.add_argument("action", choices=["start", "stop", "status"])
    d.set_defaults(func=cmd_daemon)

    i = sub.add_parser("install-claude", help="register with Claude Code (user scope)")
    i.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    i.add_argument("--dry-run", action="store_true", help="only print planned changes")
    i.add_argument("--uninstall", action="store_true")
    i.set_defaults(func=cmd_install)

    dr = sub.add_parser("doctor", help="check the environment and explain how to fix problems")
    dr.add_argument("--offline", action="store_true", help="do not contact Hugging Face")
    dr.add_argument("--json", action="store_true", help="raw JSON output")
    dr.set_defaults(func=cmd_doctor)

    dm = sub.add_parser("download-model", help="download the embedding model (asks first)")
    dm.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    dm.set_defaults(func=cmd_download_model)

    for sp in (s, st, r):
        sp.add_argument("--root", help="project root (default: CLAUDE_PROJECT_DIR / git root / cwd)")
        sp.add_argument("--json", action="store_true", help="raw JSON output")
    return p


def main(argv: list[str] | None = None) -> int:
    # previews are repository text (emoji, ✓, CJK): a piped stdout would use the ANSI code
    # page and crash on the first character it cannot encode
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DaemonUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
