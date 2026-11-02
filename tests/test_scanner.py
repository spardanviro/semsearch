from pathlib import Path

import pytest

from semsearch.config import Settings
from semsearch.index import scanner
from semsearch.index.scanner import classify, read_text, scan


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home")


def write(root: Path, rel: str, content: str | bytes = "x = 1\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(("path", "expected"), [
    ("src/a.py", ("code", "python")),
    ("Assets/Player.cs", ("code", "csharp")),
    ("README.md", ("text", "markdown")),
    ("conf/app.yaml", ("text", "yaml")),
    ("Dockerfile", ("code", "dockerfile")),
    ("Assets/Player.cs.meta", None),
    ("image.png", None),
])
def test_classify_known_and_unknown_paths_returns_expected_kind(
        path: str, expected: tuple[str, str] | None) -> None:
    assert classify(path) == expected


@pytest.mark.parametrize("lister", ["walk", "git"])
def test_scan_with_ignore_files_keeps_only_indexable_files(tmp_path: Path, settings: Settings,
                                           monkeypatch: pytest.MonkeyPatch, lister: str) -> None:
    root = tmp_path / "proj"
    write(root, "src/main.py")
    write(root, "src/gen/out.py")
    write(root, "node_modules/lib/index.js")
    write(root, "Library/cache.cs")
    write(root, ".env", "SECRET=1")
    write(root, "keys/server.pem", "-----BEGIN")
    write(root, "notes.md", "# hi\n")
    write(root, "big.json", "{" + " " * 300_000 + "}")
    write(root, "empty.py", "")
    write(root, "docs/skip.md", "# skip\n")
    write(root, ".gitignore", "src/gen/\nLibrary/\n")
    write(root, ".semsearchignore", "docs/\n")

    monkeypatch.setattr(scanner, "_list_with_rg", lambda _root: None)
    if lister == "git":
        import subprocess
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    else:
        monkeypatch.setattr(scanner, "_list_with_git", lambda _root: None)

    files = scan(root, settings)
    assert sorted(files) == ["notes.md", "src/main.py"]
    info = files["src/main.py"]
    assert info.modality == "code" and info.lang == "python" and info.size > 0


def test_scan_without_gitignore_skips_build_dirs_case_insensitively(
        tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scanner, "_list_with_rg", lambda _root: None)
    write(tmp_path, "Assets/Player.cs")
    write(tmp_path, "Library/PackageCache/pkg/Thing.cs")
    write(tmp_path, "BIN/tool.py")
    write(tmp_path, "src/build/helper.py")

    assert sorted(scan(tmp_path, settings)) == ["Assets/Player.cs"]


def test_scan_with_gitignore_keeps_source_dirs_named_like_build_output(
        tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scanner, "_list_with_rg", lambda _root: None)
    write(tmp_path, ".gitignore", "dist/\n")
    write(tmp_path, "src/build/helper.py")
    write(tmp_path, "dist/out.py")

    assert sorted(scan(tmp_path, settings)) == ["src/build/helper.py"]


def test_scan_sensitive_files_cannot_be_reincluded_by_ignore_negation(
        tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scanner, "_list_with_rg", lambda _root: None)
    write(tmp_path, "credentials.json", "{}")
    write(tmp_path, ".semsearchignore", "!credentials.json\n")
    write(tmp_path, "a.py")

    assert sorted(scan(tmp_path, settings)) == ["a.py"]


def test_scan_skips_symlink_pointing_outside_root(
        tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scanner, "_list_with_rg", lambda _root: None)
    root = tmp_path / "proj"
    write(root, "a.py")
    secret = tmp_path / "outside" / "notes.md"
    write(tmp_path, "outside/notes.md", "# private\n")
    try:
        (root / "docs.md").symlink_to(secret)
    except OSError:
        pytest.skip("creating symlinks needs developer mode on Windows")

    assert sorted(scan(root, settings)) == ["a.py"]


def test_scan_too_many_files_raises(tmp_path: Path, settings: Settings,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scanner, "_list_with_rg", lambda _root: None)
    for i in range(4):
        write(tmp_path, f"f{i}.py")
    from dataclasses import replace

    with pytest.raises(scanner.TooManyFiles, match="SEMSEARCH_MAX_FILES"):
        scan(tmp_path, replace(settings, max_files=3))


def test_stat_one_gitignored_path_returns_none(tmp_path: Path, settings: Settings) -> None:
    write(tmp_path, ".gitignore", "gen/\n")
    write(tmp_path, "gen/out.py")
    write(tmp_path, "src/a.py")

    assert scanner.stat_one(tmp_path, "gen/out.py", settings) is None
    assert scanner.stat_one(tmp_path, "src/a.py", settings) is not None


def test_stat_one_missing_or_sensitive_returns_none(tmp_path: Path, settings: Settings) -> None:
    write(tmp_path, "a.py")
    assert scanner.stat_one(tmp_path, "a.py", settings) is not None
    assert scanner.stat_one(tmp_path, "missing.py", settings) is None
    assert scanner.stat_one(tmp_path, ".env", settings) is None


def test_read_text_bom_gbk_binary_decodes_or_returns_none(tmp_path: Path) -> None:
    (tmp_path / "bom.cs").write_bytes("﻿// 注释\n".encode())
    (tmp_path / "gbk.cs").write_bytes("// 中文注释\n".encode("gbk"))
    (tmp_path / "bin.py").write_bytes(b"abc\0def")
    assert read_text(tmp_path / "bom.cs") == "// 注释\n"
    assert read_text(tmp_path / "gbk.cs") == "// 中文注释\n"
    assert read_text(tmp_path / "bin.py") is None
