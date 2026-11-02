"""Pre-build eval indexes (evals/.home) for project roots, one after another.

    uv run python evals/index_projects.py <root> [<root> ...]
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

from semsearch.config import Settings
from semsearch.embed.base import make_embedder
from semsearch.index.project import Project

HOME = Path(__file__).resolve().parent / ".home"
TIMEOUT_S = 6 * 3600


def main() -> None:
    settings = replace(Settings(), home=HOME)
    embedder = make_embedder(settings)
    for arg in sys.argv[1:]:
        t0 = time.perf_counter()
        project = Project(Path(arg), settings, embedder)
        ok = project.wait_idle(TIMEOUT_S)
        st = project.status()
        print(f"{arg}: ok={ok} files={st['files']} chunks={st['chunks']} "
              f"{time.perf_counter() - t0:.0f}s error={st['last_error']}", flush=True)
        project.close()


if __name__ == "__main__":
    main()
