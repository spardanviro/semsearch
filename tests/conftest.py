"""Test isolation: the suite must not depend on the developer's SEMSEARCH_* settings.

The README tells users to `setx SEMSEARCH_MEDIA ...` etc.; without this, such a variable
broke a dozen tests (and mcp_server builds Settings at import time, so the environment is
cleaned before any test module is imported, then again for every test).
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

ISOLATED_PREFIXES = ("SEMSEARCH_",)
ISOLATED_NAMES = ("CLAUDE_PROJECT_DIR",)
KEEP = ("SEMSEARCH_GPU_TESTS",)        # opt-in switch for the real-model tests


def _isolated(name: str) -> bool:
    return name not in KEEP and (name.startswith(ISOLATED_PREFIXES) or name in ISOLATED_NAMES)


for _name in [n for n in os.environ if _isolated(n)]:
    del os.environ[_name]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in [n for n in os.environ if _isolated(n)]:
        monkeypatch.delenv(name)
    yield
    from semsearch.index import media  # a PDF left open by a test locks the file on Windows
    media.release()
