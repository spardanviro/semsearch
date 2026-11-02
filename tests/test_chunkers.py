from dataclasses import replace

import pytest

from semsearch.config import Settings
from semsearch.index.chunkers import chunk_text


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
def small() -> Settings:
    """Tiny limits so short fixtures exercise splitting."""
    return replace(Settings(), target_tokens=60, max_tokens=120, window_lines=10,
                   window_overlap=3, preview_chars=200)


def body(n: int, word: str = "value") -> str:
    return "\n".join(f"        x{i} = {word} + {i}  # step {i} of the computation" for i in range(n))


PY = f'''import os
import sys
import json

RETRY_LIMIT = 5
TIMEOUT_SECONDS = 30
USER_AGENT = "semsearch-test"


class Small:
    def a(self):
        return 1


class Big:
    """A class large enough to be split."""

    field_one = 1
    field_two = 2
    field_three = 3

    def first(self):
{body(8)}

    def second(self):
{body(8, "other")}


@decorator
def huge():
{body(60)}
'''


def test_chunk_text_python_splits_big_class_and_keeps_small_class_whole(small: Settings) -> None:
    chunks = chunk_text("pkg/mod.py", PY, "code", "python", words, small)
    symbols = [c.symbol for c in chunks]

    assert "Small" in symbols
    assert "Big.first" in symbols and "Big.second" in symbols
    assert any(c.symbol == "huge" and c.kind.endswith("(part)") for c in chunks)
    assert any(c.symbol == "(top-level)" for c in chunks)          # imports + constant
    assert any(c.symbol == "Big" and c.kind == "declarations" for c in chunks)  # fields


def test_chunk_text_every_chunk_within_token_limit_and_has_header(small: Settings) -> None:
    chunks = chunk_text("pkg/mod.py", PY, "code", "python", words, small)

    assert chunks
    for c in chunks:
        assert c.tokens <= small.max_tokens
        assert c.text.startswith("path: pkg/mod.py\nsymbol: ")
        assert 1 <= c.start_line <= c.end_line


def test_chunk_text_decorated_function_span_includes_decorator(small: Settings) -> None:
    chunks = chunk_text("pkg/mod.py", PY, "code", "python", words, small)
    huge_parts = [c for c in chunks if c.symbol == "huge"]
    decorator_line = PY.splitlines().index("@decorator") + 1

    assert min(c.start_line for c in huge_parts) == decorator_line


CS = """using System;
using System.Collections.Generic;
using UnityEngine;

namespace Game.Player
{
    public class PlayerController : MonoBehaviour
    {
        public float speed = 5f;
        public float jumpForce = 7f;
        private Rigidbody body;

        public void Move(Vector3 direction)
        {
            body.velocity = direction * speed;
            Debug.Log("moving the player in the requested direction now");
            Debug.Log("second line to make the method a bit longer than tiny");
        }

        public void Jump()
        {
            body.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
            Debug.Log("jumping with the configured force applied upwards");
            Debug.Log("another log line so that this is not a tiny chunk");
        }
    }
}
"""


def test_chunk_text_csharp_namespace_is_transparent_and_methods_qualified(small: Settings) -> None:
    chunks = chunk_text("Assets/PlayerController.cs", CS, "code", "csharp", words, small)
    symbols = {c.symbol for c in chunks}

    assert {"PlayerController.Move", "PlayerController.Jump"} <= symbols


def test_chunk_text_import_only_gap_is_dropped(small: Settings) -> None:
    imports = "\n".join(f"using Namespace{i};" for i in range(8))
    methods = "\n".join(
        f"    public int M{i}(int x)\n    {{\n"
        + "\n".join(f"        x = x + {j}; // step {j} of method {i}" for j in range(8))
        + "\n        return x;\n    }" for i in range(4))
    src = f"{imports}\n\npublic class A\n{{\n{methods}\n}}\n"

    chunks = chunk_text("A.cs", src, "code", "csharp", words, small)

    assert chunks and not any(c.symbol == "(top-level)" for c in chunks)


def test_chunk_text_gap_starts_after_leading_imports(small: Settings) -> None:
    imports = "\n".join(f"using Namespace{i};" for i in range(6))
    fields = "\n".join(f"        int field{i} = {i}; // explains the field number {i} of the menu"
                       for i in range(6))
    methods = "\n".join(
        f"        void M{i}()\n        {{\n"
        + "\n".join(f"            Step({j}); // step {j} of method {i}" for j in range(8))
        + "\n        }" for i in range(3))
    src = f"{imports}\n\nnamespace Game\n{{\n    public class Menu\n    {{\n{fields}\n{methods}\n    }}\n}}\n"

    chunks = chunk_text("Menu.cs", src, "code", "csharp", words, small)
    gap = next(c for c in chunks if c.kind == "declarations")

    assert gap.symbol == "Menu"
    assert not gap.preview.lstrip().startswith("using")


def test_chunk_text_form_feed_keeps_line_numbers_aligned(small: Settings) -> None:
    src = "def a():\n    return 1\n\x0c\n" + "\n".join(
        f"def f{i}():\n{body(6)}\n" for i in range(4))
    lines = src.split("\n")

    chunks = chunk_text("ff.py", src, "code", "python", words, small)

    for c in chunks:
        if c.kind == "function_definition":
            name = c.symbol.rsplit(".", 1)[-1]
            assert lines[c.start_line - 1].startswith(f"def {name}(")


def test_chunk_text_deeply_nested_code_does_not_recurse_out(small: Settings) -> None:
    depth = 3000
    src = "x = " + "[" * depth + "1" + "]" * depth + "\n" + body(80)

    chunks = chunk_text("deep.py", src, "code", "python", words, small)

    assert chunks


def test_chunk_text_tiny_adjacent_methods_are_merged(small: Settings) -> None:
    methods = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(12))
    src = f"class Many:\n{methods}"
    settings = replace(small, target_tokens=40)

    chunks = chunk_text("many.py", src, "code", "python", words, settings)

    assert any(c.kind == "members" and "," in c.symbol for c in chunks)
    assert len(chunks) < 12


def test_chunk_text_doc_comment_joins_following_definition(small: Settings) -> None:
    # Regression: the comment describing a method became its own gap chunk and outranked it.
    filler = "\n".join(f"        Step({j}); // step {j} of the work" for j in range(8))
    src = ("public class Store\n{\n"
           "    // stray note separated by a blank line\n\n"
           "    /// <summary>Punish a player who left the match early.</summary>\n"
           "    // escalates on repeat offences\n"
           f"    public void Punish(int id)\n    {{\n{filler}\n    }}\n\n"
           f"    public void Other()\n    {{\n{filler}\n    }}\n}}\n")
    lines = src.split("\n")

    chunks = chunk_text("Store.cs", src, "code", "csharp", words, small)
    punish = next(c for c in chunks if c.symbol == "Store.Punish")

    assert lines[punish.start_line - 1].strip().startswith("/// <summary>")
    assert "stray note" not in punish.text
    assert not any(c.kind == "declarations" and "Punish a player" in c.text for c in chunks)


def test_chunk_text_js_comment_attaches_through_export(small: Settings) -> None:
    body_lines = "\n".join(f"  total += step{i}(items); // part {i} of planning" for i in range(8))
    src = (f"/** Split tasks into waves; throws on cycles. */\nexport function planWaves(items) {{\n"
           f"{body_lines}\n  return total;\n}}\n\nexport function other(x) {{\n{body_lines}\n}}\n")

    chunks = chunk_text("m.mjs", src, "code", "javascript", words, small)
    plan = next(c for c in chunks if c.symbol == "planWaves")

    assert plan.start_line == 1


def test_chunk_text_trailing_comment_on_code_line_is_not_moved(small: Settings) -> None:
    filler = "\n".join(f"    x{i} = {i}  # step {i} of the computation here" for i in range(8))
    src = f"def a():\n{filler}\n    return 1  # end of a\ndef b():\n{filler}\n    return 2\n"

    chunks = chunk_text("t.py", src, "code", "python", words, small)
    b = next(c for c in chunks if c.symbol == "b")

    assert src.split("\n")[b.start_line - 1].startswith("def b")


def test_chunk_text_rust_doc_comments_attach(small: Settings) -> None:
    # Regression: Rust line comments end at column 0 of the next row, so `///` never attached.
    body_rs = "\n".join(f"    let v{i} = step{i}(x); // part {i} of the work" for i in range(8))
    src = f"/// Parse the config.\n/// Second doc line.\nfn parse(x: i32) {{\n{body_rs}\n}}\n\n" \
          f"fn other(x: i32) {{\n{body_rs}\n}}\n"

    chunks = chunk_text("lib.rs", src, "code", "rust", words, small)

    assert next(c for c in chunks if c.symbol == "parse").start_line == 1


def test_chunk_text_comment_after_open_brace_keeps_class_header_out(small: Settings) -> None:
    # Regression: `{ // note` above the first method pulled the class header into it.
    filler = "\n".join(f"        Step({j}); // step {j} of the work" for j in range(8))
    src = f"public class A\n{{ // note about the class\n    public void M()\n    {{\n{filler}\n    }}\n}}\n"

    chunks = chunk_text("A.cs", src, "code", "csharp", words, small)

    assert next(c for c in chunks if c.symbol == "A.M").start_line == 3


def test_chunk_text_powershell_functions_are_named_chunks(small: Settings) -> None:
    body_ps = "\n".join(f"    $value{i} = Get-Item $path{i}  # read part {i}" for i in range(8))
    src = (f"# Ask the user for a host address and add the default port\n"
           f"function Read-HostAddress {{\n{body_ps}\n}}\n\n"
           f"function Write-Config {{\n{body_ps}\n}}\n")

    chunks = chunk_text("Start-Coop.ps1", src, "code", "powershell", words, small)
    symbols = {c.symbol for c in chunks}

    assert {"Read-HostAddress", "Write-Config"} <= symbols
    assert next(c for c in chunks if c.symbol == "Read-HostAddress").start_line == 1


def test_chunk_text_small_file_becomes_single_chunk(small: Settings) -> None:
    chunks = chunk_text("tiny.py", "def f():\n    return 1\n", "code", "python", words, small)

    assert len(chunks) == 1 and chunks[0].kind == "file" and chunks[0].symbol == "tiny.py"


def test_chunk_text_empty_file_returns_no_chunks(small: Settings) -> None:
    assert chunk_text("e.py", "\n\n  \n", "code", "python", words, small) == []


def test_chunk_text_unknown_language_falls_back_to_windows(small: Settings) -> None:
    src = "\n".join(f"line {i} with some words in it" for i in range(80))

    chunks = chunk_text("x.weird", src, "code", "no-such-lang", words, small)

    assert len(chunks) > 1 and all(c.kind == "lines" for c in chunks)
    starts = [c.start_line for c in chunks]
    assert starts == sorted(starts) and chunks[-1].end_line == 80


def test_chunk_text_markdown_splits_by_heading_and_ignores_fenced_hashes(small: Settings) -> None:
    filler = "\n".join("some descriptive prose about this part of the system" for _ in range(6))
    md = f"# Title\n{filler}\n## Install\n```bash\n# not a heading\npip install x\n```\n{filler}\n" \
         f"## Usage\n{filler}\n"

    chunks = chunk_text("README.md", md, "text", "markdown", words, small)
    symbols = [c.symbol for c in chunks]

    assert "Title > Install" in symbols and "Title > Usage" in symbols
    assert not any("not a heading" in s for s in symbols)


def test_chunk_text_windows_overlap_and_cover_all_lines(small: Settings) -> None:
    src = "\n".join(f"key{i}: value number {i}" for i in range(45))

    chunks = chunk_text("conf.yaml", src, "text", "yaml", words, small)

    covered = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    assert covered == set(range(1, 46))
    assert any(b.start_line <= a.end_line for a, b in zip(chunks, chunks[1:], strict=False))
