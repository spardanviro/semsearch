"""Security regression tests from the global audit (docs/AUDIT.md)."""
from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from semsearch import install
from semsearch.config import MODEL_ID, MODEL_REVISION, Settings
from semsearch.daemon.app import DaemonState, create_app
from semsearch.embed.base import FakeEmbedder
from semsearch.formatting import format_search, one_line
from semsearch.globs import path_filter
from semsearch.index import media
from semsearch.index.chunkers import chunk_text
from semsearch.index.scanner import is_sensitive, scan
from semsearch.paths import find_project_root, refuse_root
from semsearch.permissions import make_private


# --- untrusted patterns ----------------------------------------------------------------------
def test_semsearchignore_with_catastrophic_pattern_does_not_hang(tmp_path: Path) -> None:
    # Regression: `a*a*a*...b` against a long name backtracked for minutes and froze the daemon.
    root = tmp_path / "evil"
    root.mkdir()
    (root / ".semsearchignore").write_text("a*" * 12 + "b\n", encoding="utf-8")
    for i in range(20):
        (root / ("a" * 200 + f"{i}.py")).write_text("x = 1\n", encoding="utf-8")
    t0 = time.perf_counter()

    files = scan(root, Settings(home=tmp_path / "home"))

    assert time.perf_counter() - t0 < 10 and len(files) == 20


def test_path_glob_with_catastrophic_pattern_is_fast() -> None:
    match = path_filter(["a*" * 12 + "b"])
    assert match is not None
    t0 = time.perf_counter()

    for _ in range(200):
        match("a" * 250 + ".py")

    assert time.perf_counter() - t0 < 2


@pytest.mark.parametrize("glob,path,expected", [
    ("!tests/**", "src/a.py", True), ("!tests/**", "tests/t.py", False),
    ("./src/**", "src/x/a.py", True), ("src\\sub", "src/sub/a.py", True),
    ("**/*.{ts,tsx}", "web/a.tsx", True), ("**/*.{ts,tsx}", "web/a.js", False),
])
def test_path_globs_are_normalised(glob: str, path: str, expected: bool) -> None:
    match = path_filter([glob])
    assert match is not None and match(path) is expected


# --- secrets ---------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [
    "Credentials.json", "config/AppSettings.Production.json", "Secrets.YAML", ".ENV",
    "sub/.docker/config.json", "infra/prod.tfvars.json", "local.settings.json",
    "app-firebase-adminsdk-x1.json", "token.json", "wp-config.php", "my_password_list.txt",
])
def test_sensitive_files_are_excluded_whatever_their_case_or_depth(path: str) -> None:
    assert is_sensitive(path)


@pytest.mark.parametrize("path", ["src/tokenizer.py", "docs/auth.md", "app/settings.json"])
def test_ordinary_files_are_not_sensitive(path: str) -> None:
    assert not is_sensitive(path)


# --- roots and permissions -------------------------------------------------------------------
def test_refuse_root_blocks_home_drive_and_private_dirs(tmp_path: Path) -> None:
    home = Path.home().resolve()
    settings = Settings(home=tmp_path / "semhome")
    (tmp_path / "semhome" / "x").mkdir(parents=True)

    assert refuse_root(home, settings)
    assert refuse_root(Path(home.anchor), settings)
    assert refuse_root(home / ".claude", settings)
    assert refuse_root((tmp_path / "semhome" / "x").resolve(), settings)
    assert refuse_root(tmp_path.resolve(), settings) is None


def test_daemon_refuses_to_index_the_home_directory(tmp_path: Path) -> None:
    state = DaemonState(settings=Settings(home=tmp_path / "home", embedder="fake"),
                        embedder=FakeEmbedder())
    client = TestClient(create_app(state), base_url="http://127.0.0.1")
    client.headers["Authorization"] = f"Bearer {state.token}"

    r = client.post("/status", json={"root": str(Path.home())})

    assert r.status_code == 400 and "home directory" in r.json()["detail"]


def test_project_root_walk_stops_below_a_dotfiles_repo_in_home(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    (fake_home / ".git").mkdir(parents=True)
    work = fake_home / "work" / "notes"
    work.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)

    assert find_project_root(work) == work.resolve()


def test_non_ascii_authorization_header_is_401_not_500(tmp_path: Path) -> None:
    state = DaemonState(settings=Settings(home=tmp_path / "home", embedder="fake"),
                        embedder=FakeEmbedder())
    client = TestClient(create_app(state), base_url="http://127.0.0.1")

    r = client.get("/health", headers={b"Authorization": "Bearer ünïcode".encode()})

    assert r.status_code == 401


def _principals(path: Path) -> list[str]:
    acl = subprocess.run(["icacls", str(path)], capture_output=True, text=True, check=True).stdout
    return [ln.split(":(")[0].split()[-1] for ln in acl.splitlines()[:-2] if ":(" in ln]


@pytest.mark.skipif(os.name != "nt", reason="Windows ACLs")
def test_make_private_leaves_only_the_current_user(tmp_path: Path) -> None:
    folder = tmp_path / "private"
    token = folder / "sub" / "token.txt"
    (folder / "sub").mkdir(parents=True)
    token.write_text("secret", encoding="utf-8")
    # entries of their own (not inherited) on the folder and on a file inside it, as on CI
    # runners: a plain ACL change on the folder leaves both in place
    subprocess.run(["icacls", str(folder), "/grant", "*S-1-5-18:(OI)(CI)F", "/Q"],
                   capture_output=True, check=True)
    subprocess.run(["icacls", str(token), "/grant", "*S-1-5-18:F", "/Q"],
                   capture_output=True, check=True)

    assert make_private(folder)
    user = os.environ["USERNAME"].lower()
    for path in (folder, token):
        principals = _principals(path)
        assert principals and all(p.lower().endswith(user) for p in principals), (path, principals)


# --- output to Claude ------------------------------------------------------------------------
def test_one_line_strips_invisible_characters_and_caps_length() -> None:
    hidden = "ok\u202eevil\u200b\u2028x" + "".join(chr(0xE0000 + ord(c)) for c in "run rm")

    assert one_line(hidden) == "okevilx"
    assert len(one_line("x" * 5000)) == 201


def test_untrusted_note_covers_paths_and_symbols() -> None:
    data = {"fresh": True, "index": {}, "results": [
        {"path": "a.py", "lines": [1, 2], "symbol": "f", "kind": "k", "score": 0.5,
         "preview": "x"}]}

    note = format_search("q", data).splitlines()[1]

    assert "Paths" in note and "symbol" in note and "untrusted" in note


# --- hostile files ---------------------------------------------------------------------------
def test_deeply_nested_classes_chunk_in_bounded_time() -> None:
    # Regression: every nesting level re-counted all text inside it (depth x size).
    depth = 1500
    src = "".join(f"{'  ' * i}class C{i}:\n" for i in range(depth)) + "  " * depth + "x = 1\n"
    settings = replace(Settings(), target_tokens=40, max_tokens=80)
    t0 = time.perf_counter()

    chunks = chunk_text("deep.py", src, "code", "python", lambda t: len(t.split()), settings)

    assert chunks and time.perf_counter() - t0 < 10


def test_image_with_foreign_format_inside_is_not_decoded(tmp_path: Path) -> None:
    # Pillow sniffs content: an EPS file named .png would be handed to Ghostscript.
    fake = tmp_path / "cat.png"
    fake.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 100 100\nshowpage\n")

    with pytest.raises(Exception, match="cannot identify"):
        media.segment(fake, "image", Settings())


# --- installer and supply chain --------------------------------------------------------------
@pytest.mark.parametrize("path", ['C:/x/$(evil)/hook.exe', "C:/a`b`/hook.exe",
                                  "C:/%TEMP%/hook.exe", 'C:/q"uote/hook.exe'])
def test_shell_quote_refuses_paths_a_shell_would_expand(path: str) -> None:
    with pytest.raises(ValueError, match="refusing"):
        install.shell_quote(path)


def test_uninstall_keeps_user_hooks_sharing_our_entry() -> None:
    data = {"hooks": {"PostToolUse": [{"matcher": "Edit", "hooks": [
        {"type": "command", "command": "my-formatter"},
        {"type": "command", "command": '"C:/x/semsearch-hook.exe"'}]}]}}

    out = install.without_hook(data)

    assert out["hooks"]["PostToolUse"][0]["hooks"] == [{"type": "command",
                                                        "command": "my-formatter"}]


@pytest.fixture
def clean_revision(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("SEMSEARCH_MODEL_REVISION", raising=False)
    yield


def test_default_model_is_pinned_to_a_commit(clean_revision: None) -> None:
    assert Settings().revision == MODEL_REVISION and Settings().model == MODEL_ID
    assert replace(Settings(), model="BAAI/bge-m3").revision is None
    assert replace(Settings(), model_revision="abc").revision == "abc"
