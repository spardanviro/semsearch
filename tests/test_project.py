"""Integration tests: scanner + chunker + indexer + store + cache + search, with the fake embedder."""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from semsearch.config import Settings
from semsearch.embed.base import FakeEmbedder
from semsearch.index.project import Project

RETRY = '''def fetch_with_retry(url, attempts=3):
    """Download the url again with exponential backoff when the request fails."""
    for attempt in range(attempts):
        try:
            return http_get(url)
        except TimeoutError:
            sleep(2 ** attempt)
    raise RuntimeError("download failed after retries")
'''

RENDER = '''def render_player_mesh(player, camera):
    """Draw the player mesh with the current camera transform onto the screen."""
    mesh = player.mesh
    matrix = camera.view_projection
    draw_triangles(mesh.vertices, matrix)
'''

CONFIG = '''def load_settings(path):
    """Read the yaml configuration file and merge it with defaults."""
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return {**DEFAULTS, **data}
'''


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def bump_mtime(p: Path) -> None:
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    write(r, "net/retry.py", RETRY)
    write(r, "gfx/render.py", RENDER)
    write(r, "conf/settings.py", CONFIG)
    return r


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(Settings(home=tmp_path / "home", embedder="fake"),
                   scan_cooldown_s=0.0, search_wait_s=10.0)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def project(root: Path, settings: Settings, embedder: FakeEmbedder) -> Iterator[Project]:
    p = Project(root, settings, embedder)
    assert p.wait_idle(10)
    yield p
    p.close()


def test_project_initial_index_search_finds_relevant_file(project: Project) -> None:
    results, fresh = project.search("retry download with exponential backoff")

    assert fresh
    assert results[0].path == "net/retry.py"
    assert results[0].start_line == 1
    assert project.status()["files"] == 3


def test_project_edit_reembeds_only_changed_file(project: Project, root: Path,
                                                 embedder: FakeEmbedder) -> None:
    # Arrange
    before = embedder.documents_encoded
    p = write(root, "gfx/render.py", RENDER.replace("camera", "lens"))
    bump_mtime(p)

    # Act
    results, _ = project.search("draw mesh with lens transform")

    # Assert
    assert embedder.documents_encoded - before == 1
    assert results[0].path == "gfx/render.py"
    assert "lens" in results[0].preview
    stats = project.last_stats
    assert stats is not None and stats.files_indexed == 1 and stats.chunks_embedded == 1


def test_project_touch_without_content_change_embeds_nothing(project: Project, root: Path,
                                                             embedder: FakeEmbedder) -> None:
    before = embedder.documents_encoded
    bump_mtime(root / "net" / "retry.py")

    project.search("anything")

    assert embedder.documents_encoded == before
    assert project.last_stats is not None and project.last_stats.files_unchanged == 1


def test_project_deleted_file_disappears_from_results(project: Project, root: Path) -> None:
    (root / "net" / "retry.py").unlink()

    results, _ = project.search("retry download with exponential backoff")

    assert all(r.path != "net/retry.py" for r in results)
    assert project.status()["files"] == 2


def test_project_deleted_file_within_scan_cooldown_is_not_returned(
        root: Path, settings: Settings, embedder: FakeEmbedder) -> None:
    # Regression: a file removed outside Claude (no hook) stayed in results until the next scan.
    proj = Project(root, replace(settings, scan_cooldown_s=3600), embedder)
    try:
        assert proj.wait_idle(10)
        (root / "net" / "retry.py").unlink()

        results, _ = proj.search("retry download with exponential backoff", k=1)
        assert proj.wait_idle(10)

        assert len(results) == 1 and results[0].path != "net/retry.py"
        assert "net/retry.py" not in proj.store.files()
    finally:
        proj.close()


def test_project_notify_updates_without_full_scan(root: Path, settings: Settings,
                                                  embedder: FakeEmbedder) -> None:
    proj = Project(root, replace(settings, scan_cooldown_s=3600), embedder)
    try:
        assert proj.wait_idle(10)
        write(root, "net/cache.py", "def evict_least_recently_used(cache):\n"
                                    "    cache.popitem(last=False)\n")

        accepted = proj.notify([str(root / "net" / "cache.py"), "../outside.py"])
        assert proj.wait_idle(10)
        results, _ = proj.search("evict least recently used cache entry")

        assert accepted == ["net/cache.py"]
        assert results[0].path == "net/cache.py"
    finally:
        proj.close()


def test_project_reopen_loads_vectors_without_reembedding(root: Path, settings: Settings) -> None:
    first = Project(root, settings, FakeEmbedder())
    assert first.wait_idle(10)
    first.close()

    second_embedder = FakeEmbedder()
    second = Project(root, settings, second_embedder)
    try:
        assert second.wait_idle(10)
        results, _ = second.search("yaml configuration defaults")

        assert second_embedder.documents_encoded == 0
        assert results[0].path == "conf/settings.py"
    finally:
        second.close()


def test_project_model_change_wipes_and_rebuilds(root: Path, settings: Settings) -> None:
    first = Project(root, settings, FakeEmbedder())
    assert first.wait_idle(10)
    first.close()

    other = FakeEmbedder()
    other.model_id = "another-model"
    proj = Project(root, settings, other)
    try:
        assert proj.wait_idle(10)
        assert other.documents_encoded == 3
        assert proj.store.meta()["model_id"] == "another-model"
    finally:
        proj.close()


def test_project_full_reindex_reembeds_everything(project: Project, embedder: FakeEmbedder) -> None:
    before = embedder.documents_encoded

    project.reindex(full=True)
    assert project.wait_idle(10)

    assert embedder.documents_encoded - before == 3
    assert project.status()["chunks"] == 3


def test_project_path_glob_and_modality_filters(project: Project, root: Path) -> None:
    write(root, "docs/retry.md", "# Retry\nWe retry downloads with exponential backoff.\n")
    project.request_scan(force=True)
    assert project.wait_idle(10)

    code_only, _ = project.search("retry backoff", modality="code")
    docs_only, _ = project.search("retry backoff", path_globs=["docs/**"])

    assert all(r.modality == "code" for r in code_only)
    assert [r.path for r in docs_only] == ["docs/retry.md"]


def test_project_binary_file_is_recorded_and_not_retried(project: Project, root: Path,
                                                         embedder: FakeEmbedder) -> None:
    (root / "blob.py").write_bytes(b"\x00\x01binary")
    project.request_scan(force=True)
    assert project.wait_idle(10)
    assert project.last_stats is not None and project.last_stats.files_failed == 1

    before = embedder.documents_encoded
    project.request_scan(force=True)
    assert project.wait_idle(10)

    assert embedder.documents_encoded == before
    assert "blob.py" in project.store.files()


def test_project_identical_copies_collapse_into_one_result(project: Project, root: Path) -> None:
    write(root, "copy/net/retry.py", RETRY)
    project.request_scan(force=True)
    assert project.wait_idle(10)

    results, _ = project.search("retry download with exponential backoff")

    top = results[0]
    assert {top.path, *top.duplicates} == {"net/retry.py", "copy/net/retry.py"}
    assert sum(1 for r in results if r.symbol == top.symbol) == 1


def test_project_poison_file_falls_back_and_does_not_block_others(
        root: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: an exception while chunking one file aborted the whole update, forever.
    from semsearch.index import indexer

    real = indexer.chunk_text

    def flaky(path: str, text: str, modality: str, lang: str, *rest: object):  # type: ignore[no-untyped-def]
        if path == "gfx/render.py" and modality == "code":
            raise RecursionError("too deep")
        if path == "conf/settings.py":
            raise RuntimeError("even the fallback fails")
        return real(path, text, modality, lang, *rest)  # type: ignore[arg-type]

    monkeypatch.setattr(indexer, "chunk_text", flaky)
    proj = Project(root, settings, FakeEmbedder())
    try:
        assert proj.wait_idle(10)
        stats = proj.last_stats
        results, _ = proj.search("draw mesh camera")

        assert proj.last_error is None
        assert stats is not None and stats.files_indexed == 2 and stats.files_failed == 1
        assert results[0].path == "gfx/render.py"           # indexed through the window fallback
        assert set(proj.store.files()) == {"net/retry.py", "gfx/render.py", "conf/settings.py"}
    finally:
        proj.close()


def test_project_transient_read_error_is_retried_next_scan(
        root: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from semsearch.index import indexer

    real = indexer.read_text
    calls = {"n": 0}

    def locked_once(path: Path, limit: int | None = None) -> str | None:
        if path.name == "retry.py" and calls["n"] == 0:
            calls["n"] += 1
            raise PermissionError("file is locked by another process")
        return real(path, limit)

    monkeypatch.setattr(indexer, "read_text", locked_once)
    proj = Project(root, settings, FakeEmbedder())
    try:
        assert proj.wait_idle(10)
        assert "net/retry.py" not in proj.store.files()

        proj.request_scan(force=True)
        assert proj.wait_idle(10)

        assert "net/retry.py" in proj.store.files()
    finally:
        proj.close()


def test_project_modality_filter_finds_docs_even_when_code_dominates(
        project: Project, root: Path) -> None:
    for i in range(30):
        write(root, f"pkg/retry_{i}.py", RETRY.replace("fetch_with_retry", f"fetch_{i}"))
    write(root, "docs/notes.md", "# Notes\nA few words about the build pipeline.\n")
    project.request_scan(force=True)
    assert project.wait_idle(10)

    results, _ = project.search("retry download with exponential backoff", k=3, modality="text")

    assert [r.path for r in results] == ["docs/notes.md"]


def test_project_search_empty_query_raises(project: Project) -> None:
    with pytest.raises(ValueError):
        project.search("   ")


def test_project_scan_cooldown_skips_rapid_rescans(root: Path, settings: Settings) -> None:
    proj = Project(root, replace(settings, scan_cooldown_s=3600), FakeEmbedder())
    try:
        assert proj.wait_idle(10)
        assert proj.request_scan() is False
        assert proj.request_scan(force=True) is True
    finally:
        proj.close()


def test_project_status_reports_progress_and_paths(project: Project) -> None:
    status = project.status()

    assert status["indexing"] is False
    assert status["progress"] == {"done": 3, "total": 3}
    assert str(status["index_path"]).endswith("index.db")
    assert status["last_error"] is None
    assert isinstance(status["last_scan_at"], float) and status["last_scan_at"] <= time.time()
