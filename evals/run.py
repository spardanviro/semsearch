"""Retrieval quality benchmark.

    uv run python evals/run.py                       # all case files, default modes
    uv run python evals/run.py --modes vector,hybrid --show-misses

Each case file names a project root; that project is indexed into an isolated home
(evals/.home) with the real embedder, so the running daemon's index is never touched.
A result is a hit when its path (or one of its identical copies) matches an expectation and
the expected symbol name or line range matches too. Reports Recall@1/5/10 and MRR@10.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from semsearch.config import Settings
from semsearch.embed.base import make_embedder
from semsearch.index.project import Project
from semsearch.index.search import SearchResult, search

HERE = Path(__file__).resolve().parent
CUTOFFS = (1, 5, 10)
TOP_K = max(CUTOFFS)
INDEX_TIMEOUT_S = 1800
PRIVATE = HERE / "private"   # gitignored: cases about projects that must stay local


def _load_translations() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in (HERE / "cases" / "translations_en.json", PRIVATE / "translations_en.json"):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            out.update({k: v for k, v in data.items() if not k.startswith("_")})
    return out


TRANSLATIONS = _load_translations()


@dataclass(frozen=True)
class Case:
    suite: str
    query: str
    tags: tuple[str, ...]
    expect: tuple[dict[str, object], ...]


def load_suite(path: Path) -> tuple[Path, tuple[str, ...], list[Case]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = [Case(path.stem, c["q"], tuple(c.get("tags", ())), tuple(c["expect"]))
             for c in data["cases"]]
    # "~/..." and $VARS are expanded; relative paths are taken from the repository root
    root = Path(os.path.expandvars(os.path.expanduser(data["project"])))
    if not root.is_absolute():
        root = (HERE.parent / root).resolve()
    return root, tuple(data.get("aliases", ())), cases


def _norm(path: str, aliases: tuple[str, ...]) -> str:
    for prefix in aliases:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


Locator = Callable[[str, str], "int | None"]
_COMMENT_LINE = re.compile(r"^\s*(//|#|/\*|\*|--|;)")
_DECL_WORDS = ("def|function|func|fn|class|struct|interface|enum|record|void|static|public|"
               "private|protected|internal|override|async|export|const|let|var|bool|int|"
               "string|float|double|Task|IEnumerable")


def definition_locator(root: Path) -> Locator:
    """(path, symbol) -> 1-based line of the symbol's first non-comment mention (its
    declaration in practice), so a hit on a chunk that merely *contains* the definition (a
    whole small file, a small class) still counts. Approximate, but only ever used as a
    fallback after the exact symbol-name match."""
    cache: dict[tuple[str, str], int | None] = {}

    def locate(path: str, symbol: str) -> int | None:
        key = (path, symbol)
        if key not in cache:
            name = rf"(?<![\w.]){re.escape(symbol)}(?![\w-])"
            # prefer a declaration line; fall back to the first non-comment mention
            declared = re.compile(rf"\b(?:{_DECL_WORDS})\b.*{name}")
            mention = re.compile(name)
            try:
                lines = (root / path).read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                lines = []
            code = [(i, ln) for i, ln in enumerate(lines, 1) if not _COMMENT_LINE.match(ln)]
            cache[key] = (next((i for i, ln in code if declared.search(ln)), None)
                          or next((i for i, ln in code if mention.search(ln)), None))
        return cache[key]
    return locate


def matches(result: SearchResult, exp: dict[str, object], aliases: tuple[str, ...],
            locate: Locator | None = None) -> bool:
    want = _norm(str(exp["path"]), aliases)
    paths = {_norm(p, aliases) for p in (result.path, *result.duplicates)}
    if want not in paths:
        return False
    symbol, lines = exp.get("symbol"), exp.get("lines")
    if symbol is None and lines is None:
        return True
    if symbol is not None and str(symbol) in re.split(r"[.,\s]+", result.symbol):
        return True
    if isinstance(lines, list) and result.page is not None:      # PDF: lines = page range
        return int(lines[0]) <= result.page <= int(lines[1])
    if result.start_line is None or result.end_line is None:
        return False
    if isinstance(lines, list):
        return result.start_line <= int(lines[1]) and int(lines[0]) <= result.end_line
    if symbol is not None and locate is not None:
        line = locate(str(exp["path"]), str(symbol))
        return line is not None and result.start_line <= line <= result.end_line
    return False


def first_hit(results: list[SearchResult], case: Case, aliases: tuple[str, ...],
              locate: Locator | None = None) -> int | None:
    for rank, r in enumerate(results, 1):
        if any(matches(r, e, aliases, locate) for e in case.expect):
            return rank
    return None


def run_query(project: Project, query: str, mode: str) -> list[SearchResult]:
    """`mode` is `<vector|keyword|hybrid>[@<keyword weight>][;pf=<per file>][;tr=1]`.

    tr=1: replace the query by its English version from cases/translations_en.json, if any.
    """
    head, *opts = mode.split(";")
    name, _, weight = head.partition("@")
    options = dict(o.split("=", 1) for o in opts)
    if options.get("tr") == "1":
        query = TRANSLATIONS.get(query, query)
    return search(project.store, project.cache, project.embedder, project.db_lock, query, TOP_K,
                  per_file=int(options.get("pf") or project.settings.per_file_results),
                  mode=name,
                  keyword_weight=float(weight) if weight else project.settings.keyword_weight)


def unindexed_targets(project: Project, cases: list[Case]) -> list[tuple[str, str]]:
    """Expectations naming a file the index does not hold (typo, ignored or too large):
    such a case can never be hit, so it would silently count as a search failure."""
    with project.db_lock:
        indexed = set(project.store.files())
    return [(c.query, str(e["path"])) for c in cases for e in c.expect
            if str(e["path"]) not in indexed]


def summarize(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks) or 1
    out = {f"R@{c}": sum(1 for r in ranks if r is not None and r <= c) / n for c in CUTOFFS}
    out["MRR"] = sum(1 / r for r in ranks if r is not None and r <= TOP_K) / n
    return out


def report(rows: list[dict[str, object]], modes: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    groups: dict[str, list[dict[str, object]]] = {"all": rows}
    for row in rows:
        groups.setdefault(f"suite:{row['suite']}", []).append(row)
        for tag in row["tags"]:  # type: ignore[union-attr]
            groups.setdefault(f"tag:{tag}", []).append(row)
    table: dict[str, dict[str, dict[str, float]]] = {}
    header = f"{'group':<18}{'n':>4}  " + "  ".join(
        f"{m + ' ' + k:>14}" for m in modes for k in ("R@1", "R@5", "MRR"))
    print(header)
    for name, items in groups.items():
        table[name] = {m: summarize([it["ranks"][m] for it in items]) for m in modes}  # type: ignore[index]
        cells = "  ".join(f"{table[name][m][k]:>14.3f}" for m in modes for k in ("R@1", "R@5", "MRR"))
        print(f"{name:<18}{len(items):>4}  {cells}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="*", type=Path)
    ap.add_argument("--modes", default="vector")
    ap.add_argument("--home", type=Path, default=HERE / ".home")
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="fail when an expected path is not in the index")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "latest.json")
    args = ap.parse_args()
    modes = args.modes.split(",")
    suites = args.suites or sorted(p for d in (HERE / "cases", PRIVATE / "cases")
                                   for p in d.glob("*.json")
                                   if not p.name.startswith("translations"))
    settings = replace(Settings(), home=args.home)
    embedder = make_embedder(settings)

    rows: list[dict[str, object]] = []
    for suite_path in suites:
        root, aliases, cases = load_suite(suite_path)
        if not root.is_dir():   # e.g. a suite about a project that only exists on its author's disk
            print(f"[{suite_path.stem}] skipped: project {root} not found", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        project = Project(root, settings, embedder)
        if not project.wait_idle(INDEX_TIMEOUT_S):
            raise SystemExit(f"indexing {root} did not finish")
        print(f"[{suite_path.stem}] indexed in {time.perf_counter() - t0:.1f}s "
              f"{project.status()['chunks']} chunks", file=sys.stderr)
        unknown = unindexed_targets(project, cases)
        for query, path in unknown:
            print(f"  ! expected path not in index: {path}  ({query})", file=sys.stderr)
        if unknown and args.strict:
            raise SystemExit(f"{suite_path.name}: {len(unknown)} expected paths are not indexed")
        locate = definition_locator(root)
        for case in cases:
            ranks: dict[str, int | None] = {}
            for mode in modes:
                ranks[mode] = first_hit(run_query(project, case.query, mode), case, aliases,
                                        locate)
            rows.append({"suite": case.suite, "query": case.query, "tags": list(case.tags),
                         "ranks": ranks})
        project.close()

    table = report(rows, modes)
    if args.show_misses:
        print("\nper-case ranks (- = not in top 10):")
        for row in rows:
            cells = " ".join(f"{m}={row['ranks'][m] or '-'}" for m in modes)  # type: ignore[index]
            print(f"  {cells:<28} [{row['suite']}] {row['query']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"modes": modes, "summary": table, "cases": rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
