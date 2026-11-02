"""Retrieval-quality regression tests from the global audit (docs/AUDIT.md)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from semsearch.config import Settings
from semsearch.embed.base import FakeEmbedder
from semsearch.index import keywords
from semsearch.index.chunkers import chunk_text, looks_minified
from semsearch.index.project import Project
from semsearch.index.scanner import read_text


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
def small() -> Settings:
    return replace(Settings(), target_tokens=60, max_tokens=120, window_lines=10,
                   window_overlap=3, preview_chars=200)


def body(n: int, indent: str = "    ") -> str:
    return "\n".join(f"{indent}step{i} := compute(value, {i}) // part {i} of the work"
                     for i in range(n))


# --- long lines ------------------------------------------------------------------------------
def test_single_overlong_line_is_split_into_chunks_within_the_limit(small: Settings) -> None:
    # Regression: one 6000-token line became one chunk; the model only saw its first part.
    long_line = " ".join(f"item{i}" for i in range(2000))
    src = f"def head():\n    return 1\n{long_line}\ndef tail():\n    return 2\n" + "x = 1\n" * 40

    chunks = chunk_text("data.py", src, "code", "python", words, small)
    parts = [c for c in chunks if c.start_line == 3]

    assert len(parts) > 10 and all(c.tokens <= small.max_tokens for c in chunks)
    assert "item1999" in parts[-1].text


def test_minified_bundles_are_not_indexed() -> None:
    bundle = ";".join(f"function f{i}(a){{return a+{i}}}" for i in range(3000))
    nested = "".join(f"{'  ' * i}class C{i}:\n" for i in range(800))

    assert looks_minified(bundle, 1)
    assert not looks_minified(nested, 800)


# --- symbol names ----------------------------------------------------------------------------
def test_go_type_declarations_are_named(small: Settings) -> None:
    # Regression: every Go `type X struct` was named <anonymous>.
    src = (f"package main\n\ntype Server struct {{\n{body(12)}\n}}\n\n"
           f"type Handler interface {{\n{body(12)}\n}}\n\nfunc main() {{\n{body(12)}\n}}\n")

    symbols = {c.symbol for c in chunk_text("main.go", src, "code", "go", words, small)}

    assert {"Server", "Handler", "main"} <= symbols and "<anonymous>" not in symbols


def test_rust_generic_impl_blocks_are_named(small: Settings) -> None:
    fn_body = "\n".join(f"        let v{i} = step{i}(x); // part {i}" for i in range(10))
    src = (f"impl<T> Stack<T> {{\n    fn push(&mut self, x: T) {{\n{fn_body}\n    }}\n"
           f"    fn pop(&mut self) {{\n{fn_body}\n    }}\n}}\n")

    symbols = {c.symbol for c in chunk_text("lib.rs", src, "code", "rust", words, small)}

    assert {"Stack.push", "Stack.pop"} <= symbols


def test_js_arrow_function_constants_are_definitions(small: Settings) -> None:
    # Regression: `export const useThing = () => {}` landed in an unnamed gap chunk.
    inner = "\n".join(f"  const v{i} = compute(items, {i}); // step {i}" for i in range(10))
    src = (f"/** Load the user profile. */\nexport const useProfile = (id) => {{\n{inner}\n}};\n\n"
           f"const helper = function (x) {{\n{inner}\n}};\n\nexport const LIMIT = 5;\n")

    chunks = chunk_text("hooks.ts", src, "code", "typescript", words, small)
    prof = next(c for c in chunks if c.symbol == "useProfile")

    assert prof.kind == "function" and prof.start_line == 1
    assert "helper" in {c.symbol for c in chunks}


def test_rust_doc_comment_survives_attributes(small: Settings) -> None:
    fn_body = "\n".join(f"    let v{i} = step{i}(x); // part {i}" for i in range(10))
    src = (f"/// Parse the header.\n#[inline]\n#[must_use]\nfn parse(x: u8) {{\n{fn_body}\n}}\n\n"
           f"fn other(x: u8) {{\n{fn_body}\n}}\n")

    chunks = chunk_text("p.rs", src, "code", "rust", words, small)

    assert next(c for c in chunks if c.symbol == "parse").start_line == 1


# --- markdown --------------------------------------------------------------------------------
def test_markdown_heading_keeps_a_trailing_hash_and_nested_fences(small: Settings) -> None:
    filler = "\n".join("some descriptive prose about this part of the system" for _ in range(6))
    md = (f"# Learn C#\n{filler}\n````md\n```bash\n# not a heading\n```\n````\n{filler}\n"
          f"## Next ##\n{filler}\n")

    symbols = [c.symbol for c in chunk_text("g.md", md, "text", "markdown", words, small)]

    assert "Learn C#" in symbols and "Learn C# > Next" in symbols
    assert not any("not a heading" in s for s in symbols)


# --- files and keywords ----------------------------------------------------------------------
def test_utf16_files_with_bom_are_read(tmp_path: Path) -> None:
    # Regression: PowerShell 5 / SSMS output (UTF-16) was dropped as binary.
    p = tmp_path / "script.ps1"
    p.write_bytes("﻿Write-Host 'héllo'\r\n".encode("utf-16-le"))

    text = read_text(p)

    assert text is not None and "héllo" in text


def test_keyword_terms_skip_digit_prefixed_words() -> None:
    assert keywords.terms("the 3rd tab is 10px wide") == ["the", "tab", "is", "wide"]


# --- search ----------------------------------------------------------------------------------
def test_many_identical_copies_do_not_crowd_out_other_results(tmp_path: Path) -> None:
    # Regression: 70 vendored copies filled the candidate pool; k=8 returned a single result.
    root = tmp_path / "proj"
    code = "def fetch_with_retry(url):\n    return download(url, retries=3)\n"
    for i in range(70):
        (root / "vendor" / f"v{i}").mkdir(parents=True)
        (root / "vendor" / f"v{i}" / "retry.py").write_text(code, encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "other.py").write_text("def fetch_page(url):\n    return get(url)\n", "utf-8")
    settings = replace(Settings(home=tmp_path / "home", embedder="fake"), search_wait_s=10.0)
    p = Project(root, settings, FakeEmbedder())
    try:
        assert p.wait_idle(30)
        results, _ = p.search("fetch url with retry download", k=3)
        assert {r.path for r in results} >= {"src/other.py"}
        assert len(results[0].duplicates) == 69
    finally:
        p.close()


def test_settings_reject_empty_home_and_non_positive_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMSEARCH_HOME", "")
    assert Settings().home.name == "semsearch"                 # "" no longer means the cwd
    monkeypatch.setenv("SEMSEARCH_MAX_FILES", "0")
    with pytest.raises(ValueError, match="SEMSEARCH_MAX_FILES"):
        Settings()
