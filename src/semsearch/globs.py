"""gitignore-style matchers built from untrusted patterns (.semsearchignore, path_glob).

pathspec's default backend is Python `re`, which backtracks exponentially on patterns such as
`a*a*a*a*a*a*b` against long file names: a malicious repository could freeze the daemon. With
google-re2 installed (a dependency) pathspec matches in linear time; if RE2 is unavailable on
a platform, patterns with many wildcards are dropped instead.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable

import pathspec

log = logging.getLogger(__name__)
MAX_LINES = 1000
MAX_LINE_CHARS = 300
MAX_WILDCARDS_WITHOUT_RE2 = 3
MAX_BRACE_EXPANSIONS = 64
_BRACES = re.compile(r"\{([^{}]*)\}")


def _has_re2(spec: pathspec.PathSpec) -> bool:
    backend = getattr(spec, "_backend", None)
    return type(backend).__name__.lower().startswith("re2")


def safe_spec(lines: Iterable[str], source: str) -> pathspec.PathSpec:
    """PathSpec from untrusted lines: size-capped, and wildcard-capped without RE2."""
    kept = []
    for i, line in enumerate(lines):
        if i >= MAX_LINES:
            log.warning("%s: only the first %d patterns are used", source, MAX_LINES)
            break
        if len(line) > MAX_LINE_CHARS:
            log.warning("%s: pattern longer than %d characters ignored", source, MAX_LINE_CHARS)
            continue
        kept.append(line)
    spec = pathspec.PathSpec.from_lines("gitignore", kept)
    if _has_re2(spec):
        return spec
    safe = [ln for ln in kept if len(re.findall(r"\*+", ln)) <= MAX_WILDCARDS_WITHOUT_RE2]
    if len(safe) < len(kept):
        log.warning("%s: %d patterns with many wildcards ignored (google-re2 unavailable)",
                    source, len(kept) - len(safe))
    return pathspec.PathSpec.from_lines("gitignore", safe)


def expand_braces(glob: str) -> list[str]:
    """`*.{ts,tsx}` -> `*.ts`, `*.tsx` (gitignore syntax has no braces)."""
    out = [glob]
    while any(_BRACES.search(g) for g in out) and len(out) <= MAX_BRACE_EXPANSIONS:
        nxt = []
        for g in out:
            m = _BRACES.search(g)
            if m is None:
                nxt.append(g)
                continue
            nxt.extend(g[:m.start()] + alt + g[m.end():] for alt in m.group(1).split(","))
        out = nxt
    return out[:MAX_BRACE_EXPANSIONS]


def normalize_globs(globs: Iterable[str]) -> list[str]:
    """Forgive the ways people write path filters: `./src`, `src\\sub`, `*.{py,md}`, and
    exclude-only lists (`!tests/**` alone would otherwise match nothing)."""
    out: list[str] = []
    for raw in globs:
        g = raw.strip().replace("\\", "/")
        neg = g.startswith("!")
        g = g[1:] if neg else g
        while g.startswith("./"):
            g = g[2:]
        if not g:
            continue
        out.extend(("!" if neg else "") + e for e in expand_braces(g))
    if out and all(g.startswith("!") for g in out):
        out.insert(0, "**")
    return out


def path_filter(globs: list[str] | None) -> Callable[[str], bool] | None:
    if not globs:
        return None
    spec = safe_spec(normalize_globs(globs), "path_glob")
    return spec.match_file
