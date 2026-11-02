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
import re
import sys
import time
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
TRANSLATIONS: dict[str, str] = {
    k: v for k, v in json.loads((HERE / "cases" / "translations_en.json").read_text(
        encoding="utf-8")).items() if not k.startswith("_")}


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
    return Path(data["project"]), tuple(data.get("aliases", ())), cases


def _norm(path: str, aliases: tuple[str, ...]) -> str:
    for prefix in aliases:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def matches(result: SearchResult, exp: dict[str, object], aliases: tuple[str, ...]) -> bool:
    want = _norm(str(exp["path"]), aliases)
    paths = {_norm(p, aliases) for p in (result.path, *result.duplicates)}
    if want not in paths:
        return False
    symbol, lines = exp.get("symbol"), exp.get("lines")
    if symbol is None and lines is None:
        return True
    if symbol is not None and str(symbol) in re.split(r"[.,\s]+", result.symbol):
        return True
    if isinstance(lines, list) and result.start_line is not None and result.end_line is not None:
        return result.start_line <= int(lines[1]) and int(lines[0]) <= result.end_line
    return False


def first_hit(results: list[SearchResult], case: Case, aliases: tuple[str, ...]) -> int | None:
    for rank, r in enumerate(results, 1):
        if any(matches(r, e, aliases) for e in case.expect):
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
    ap.add_argument("--out", type=Path, default=HERE / "results" / "latest.json")
    args = ap.parse_args()
    modes = args.modes.split(",")
    suites = args.suites or sorted(p for p in (HERE / "cases").glob("*.json")
                                   if not p.name.startswith("translations"))
    settings = replace(Settings(), home=args.home)
    embedder = make_embedder(settings)

    rows: list[dict[str, object]] = []
    for suite_path in suites:
        root, aliases, cases = load_suite(suite_path)
        t0 = time.perf_counter()
        project = Project(root, settings, embedder)
        if not project.wait_idle(INDEX_TIMEOUT_S):
            raise SystemExit(f"indexing {root} did not finish")
        print(f"[{suite_path.stem}] indexed in {time.perf_counter() - t0:.1f}s "
              f"{project.status()['chunks']} chunks", file=sys.stderr)
        for case in cases:
            ranks: dict[str, int | None] = {}
            for mode in modes:
                ranks[mode] = first_hit(run_query(project, case.query, mode), case, aliases)
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
