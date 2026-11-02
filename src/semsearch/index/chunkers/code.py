"""Syntax-aware code chunking with tree-sitter.

Rules
- functions/methods are chunks; a class that fits `target_tokens` stays whole,
  a bigger one is split into its members
- definitions over `max_tokens` are split with line windows
- lines not covered by any definition (imports, fields, globals) are windowed
- adjacent tiny chunks of the same parent are merged
Node type names are shared across most tree-sitter grammars, so one rule set serves all languages.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

from semsearch.index.chunkers.base import Chunk, ChunkContext
from semsearch.index.chunkers.window import window_chunks

log = logging.getLogger(__name__)

LEAF_TYPES = frozenset({
    "function_definition", "function_declaration", "generator_function_declaration",
    "method_definition", "method_declaration", "constructor_declaration",
    "destructor_declaration", "operator_declaration", "conversion_operator_declaration",
    "function_item", "method", "singleton_method", "constructor_definition",
    "type_declaration", "struct_item", "enum_item", "enum_declaration",
    "function_statement",   # PowerShell
})
CONTAINER_TYPES = frozenset({
    "class_definition", "class_declaration", "abstract_class_declaration",
    "struct_declaration", "interface_declaration", "record_declaration",
    "record_struct_declaration", "impl_item", "trait_item", "mod_item", "class_specifier",
    "struct_specifier", "object_declaration", "protocol_declaration", "class", "module",
})
NAME_TYPES = ("identifier", "type_identifier", "property_identifier", "field_identifier",
              "name", "constant", "simple_identifier", "function_name")
VAR_DECL_TYPES = frozenset({"lexical_declaration", "variable_declaration"})   # JS / TS
FUNCTION_VALUES = frozenset({"arrow_function", "function_expression", "function",
                             "generator_function"})
ATTRIBUTE_TYPES = frozenset({"attribute_item", "inner_attribute_item"})       # Rust
MIN_GAP_NON_BLANK = 3
MIN_GAP_CHARS = 80
MAX_CONTAINER_DEPTH = 30
PARSER_RETRY_S = 300.0
GAP_KIND = "declarations"
SMALL_CHUNK_TOKENS = 100
# lines that carry no meaning on their own: imports, package/namespace lines, lone braces
_BOILERPLATE = re.compile(
    r"^\s*(?:using\s|import\s|from\s+\S+\s+import\s|#\s*include\b|package\s|extern\s+crate\s"
    r"|use\s|namespace\s|require\(|[{}()\[\];,]+\s*$)")
_local = threading.local()   # tree-sitter parsers are not thread-safe: one set per thread


def _parser(lang: str) -> Parser | None:
    """Per-thread parser cache. A failure is retried after PARSER_RETRY_S: grammars are
    downloaded on first use, so an offline moment must not mean "windows forever"."""
    parsers: dict[str, tuple[Parser | None, float]] = getattr(_local, "parsers", None) or {}
    _local.parsers = parsers
    cached = parsers.get(lang)
    if cached is not None and (cached[0] is not None
                               or time.monotonic() - cached[1] < PARSER_RETRY_S):
        return cached[0]
    try:
        parsers[lang] = (get_parser(lang), time.monotonic())  # type: ignore[arg-type]
    except Exception as exc:  # unknown, offline or broken grammar -> caller uses windows
        log.warning("no tree-sitter parser for %s (line windows instead): %s", lang, exc)
        parsers[lang] = (None, time.monotonic())
    return parsers[lang][0]


def _meaningful(lines: list[str], start: int, end: int) -> int:
    return sum(1 for ln in lines[start - 1:end] if ln.strip() and not _BOILERPLATE.match(ln))


def _meaningful_chars(lines: list[str], start: int, end: int) -> int:
    return sum(len(ln.strip()) for ln in lines[start - 1:end]
               if ln.strip() and not _BOILERPLATE.match(ln))


def _function_declarator(node: Node) -> Node | None:
    """`const f = () => {}` / `var f = function () {}`: the declarator holding the name."""
    if node.type not in VAR_DECL_TYPES:
        return None
    for decl in node.named_children:
        value = decl.child_by_field_name("value") if decl.type == "variable_declarator" else None
        if value is not None and value.type in FUNCTION_VALUES:
            return decl
    return None


def _text(node: Node | None) -> str | None:
    return node.text.decode("utf-8", "replace") if node is not None and node.text else None


def _special_name(node: Node) -> str | None:
    if node.type == "type_declaration":               # Go: the name sits in a type_spec
        for spec in node.named_children:
            name = _text(spec.child_by_field_name("name"))
            if name:
                return name
    if node.type == "impl_item":                      # Rust: impl<T> Display for S<T>
        typ = node.child_by_field_name("type")
        if typ is not None:
            ident = typ if typ.type == "type_identifier" else next(
                (c for c in typ.named_children if c.type == "type_identifier"), typ)
            return _text(ident)
    decl = _function_declarator(node)
    if decl is not None:
        return _text(decl.child_by_field_name("name"))
    return None


def _node_name(node: Node) -> str:
    special = _special_name(node)
    if special:
        return special
    target = node.child_by_field_name("definition") or node  # python decorated_definition
    named = target.child_by_field_name("name")
    if named is None:
        declarator = target.child_by_field_name("declarator")
        stack = [declarator] if declarator is not None else list(target.named_children)
        while stack and named is None:
            cur = stack.pop(0)
            if cur.type in NAME_TYPES:
                named = cur
            elif declarator is not None:
                stack.extend(cur.named_children)
    if named is None or named.text is None:
        return "<anonymous>"
    return named.text.decode("utf-8", "replace")


def _kind(node: Node) -> str:
    if _function_declarator(node) is not None:
        return "function"
    inner = node.child_by_field_name("definition")
    return (inner or node).type


def _last_row(node: Node) -> int:
    """Row of a node's last character: some grammars (Rust `///`) end a line comment at
    column 0 of the next row."""
    row, col = node.end_point
    return row - 1 if col == 0 and row > node.start_point[0] else row


def _leading_comment_row(node: Node, lines: list[str]) -> int:
    """0-based first row of the comment block directly above `node` (its doc comment), or
    the node's own start row. The block must touch the definition (no blank line between)
    and each comment must start its line: `} // end` or `class A { // note` stay put."""
    start = node.start_point[0]
    # `export function f` / `public` wrappers: the comment is a sibling of the wrapper
    anchor = node
    while (anchor.parent is not None and anchor.prev_named_sibling is None
           and anchor.parent.start_point[0] == start and anchor.parent.parent is not None):
        anchor = anchor.parent
    sibling = anchor.prev_named_sibling
    while (sibling is not None and ("comment" in sibling.type or sibling.type in ATTRIBUTE_TYPES)
           and _last_row(sibling) == start - 1):
        row, col = sibling.start_point          # col is a byte offset
        if row >= len(lines) or lines[row].encode("utf-8")[:col].strip():
            break
        start = row              # Rust `#[derive]` between `///` and the item belongs to it
        sibling = sibling.prev_named_sibling
    return start


@dataclass
class _Walk:
    ctx: ChunkContext
    lines: list[str]
    chunks: list[tuple[Chunk, str]] = field(default_factory=list)   # (chunk, parent qual)
    containers: list[tuple[int, int, str]] = field(default_factory=list)

    def emit_def(self, node: Node, parent: str, stack: list[tuple[Node, str]]) -> None:
        # the doc comment above a definition describes it best: keep them in one chunk
        start = _leading_comment_row(node, self.lines) + 1
        # a node ending at column 0 of the next row does not really occupy that row
        end = node.end_point[0] + (0 if node.end_point[1] == 0 and node.end_point[0] > 0 else 1)
        end = min(max(end, start), len(self.lines))
        name = _node_name(node)
        qual = f"{parent}.{name}" if parent else name
        kind = _kind(node)
        chunk = self.ctx.build(self.lines, start, end, qual, kind)
        # each nesting level re-counts the tokens of everything inside it: past a sane depth a
        # (hostile or generated) file would cost depth x size, so deeper classes are windowed
        nested = qual.count(".")
        if (kind in CONTAINER_TYPES and chunk.tokens > self.ctx.target_tokens
                and nested < MAX_CONTAINER_DEPTH):
            self.containers.append((start, end, qual))
            stack.append((node, qual))
        elif chunk.tokens > self.ctx.max_tokens:
            parts = window_chunks(self.ctx, self.lines, start, end, qual, kind + " (part)")
            self.chunks.extend((p, parent) for p in parts)
        else:
            self.chunks.append((chunk, parent))

    def visit(self, root: Node) -> None:
        """Iterative (not recursive): deeply nested generated code must not hit RecursionError."""
        stack: list[tuple[Node, str]] = [(root, "")]
        while stack:
            node, parent = stack.pop()
            for child in node.named_children:
                kind = _kind(child)
                if kind in LEAF_TYPES or kind in CONTAINER_TYPES or kind == "function":
                    self.emit_def(child, parent, stack)
                elif child.named_child_count:
                    stack.append((child, parent))


def _gap_chunks(walk: _Walk, total: int) -> list[tuple[Chunk, str]]:
    covered = bytearray(total + 2)
    for chunk, _ in walk.chunks:
        covered[chunk.start_line:chunk.end_line + 1] = b"\1" * (chunk.end_line - chunk.start_line + 1)
    out: list[tuple[Chunk, str]] = []
    line = 1
    while line <= total:
        if covered[line]:
            line += 1
            continue
        start = line
        while line <= total and not covered[line]:
            line += 1
        end = line - 1
        # import/using blocks rank high on file-name similarity but carry no information; a
        # short gap still counts when it holds real text (module docstring, documented
        # constants, a long data line)
        if (_meaningful(walk.lines, start, end) < MIN_GAP_NON_BLANK
                and _meaningful_chars(walk.lines, start, end) < MIN_GAP_CHARS):
            continue
        # trim leading imports/braces and blank edges: ownership becomes exact and the
        # preview starts with the meaningful part instead of a wall of `using` lines
        while not walk.lines[start - 1].strip() or _BOILERPLATE.match(walk.lines[start - 1]):
            start += 1
        while not walk.lines[end - 1].strip():
            end -= 1
        owner = next((q for s, e, q in reversed(walk.containers) if s <= start and end <= e), "")
        symbol = owner or "(top-level)"
        for part in window_chunks(walk.ctx, walk.lines, start, end, symbol, GAP_KIND):
            out.append((part, owner))
    return out


def _merge_small(ctx: ChunkContext, lines: list[str],
                 items: list[tuple[Chunk, str]]) -> list[Chunk]:
    out: list[tuple[Chunk, str]] = []
    for chunk, parent in sorted(items, key=lambda it: (it[0].start_line, it[0].end_line)):
        if out:
            prev, prev_parent = out[-1]
            if (prev_parent == parent and GAP_KIND not in (prev.kind, chunk.kind)
                    and prev.tokens < SMALL_CHUNK_TOKENS
                    and chunk.tokens < SMALL_CHUNK_TOKENS
                    and 0 <= chunk.start_line - prev.end_line <= 3):
                merged = ctx.build(lines, prev.start_line, chunk.end_line,
                                   f"{prev.symbol}, {chunk.symbol.rsplit('.', 1)[-1]}", "members")
                if merged.tokens <= ctx.target_tokens:
                    out[-1] = (merged, parent)
                    continue
        out.append((chunk, parent))
    return [c for c, _ in out]


def code_chunks(ctx: ChunkContext, lines: list[str], source: bytes) -> list[Chunk] | None:
    """None when the language has no usable parser (caller falls back to windows)."""
    parser = _parser(ctx.lang)
    if parser is None:
        return None
    tree = parser.parse(source)
    walk = _Walk(ctx, lines)
    walk.visit(tree.root_node)
    walk.chunks.extend(_gap_chunks(walk, len(lines)))
    return _merge_small(ctx, lines, walk.chunks)
