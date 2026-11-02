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
})
CONTAINER_TYPES = frozenset({
    "class_definition", "class_declaration", "abstract_class_declaration",
    "struct_declaration", "interface_declaration", "record_declaration",
    "record_struct_declaration", "impl_item", "trait_item", "mod_item", "class_specifier",
    "struct_specifier", "object_declaration", "protocol_declaration", "class", "module",
})
NAME_TYPES = ("identifier", "type_identifier", "property_identifier", "field_identifier",
              "name", "constant", "simple_identifier")
MIN_GAP_NON_BLANK = 3
GAP_KIND = "declarations"
SMALL_CHUNK_TOKENS = 100
# lines that carry no meaning on their own: imports, package/namespace lines, lone braces
_BOILERPLATE = re.compile(
    r"^\s*(?:using\s|import\s|from\s+\S+\s+import\s|#\s*include\b|package\s|extern\s+crate\s"
    r"|use\s|namespace\s|require\(|[{}()\[\];,]+\s*$)")
_local = threading.local()   # tree-sitter parsers are not thread-safe: one set per thread


def _parser(lang: str) -> Parser | None:
    parsers: dict[str, Parser | None] = getattr(_local, "parsers", None) or {}
    _local.parsers = parsers
    if lang not in parsers:
        try:
            parsers[lang] = get_parser(lang)  # type: ignore[arg-type]
        except Exception as exc:  # unknown or broken grammar -> caller falls back to windows
            log.info("no tree-sitter parser for %s: %s", lang, exc)
            parsers[lang] = None
    return parsers[lang]


def _meaningful(lines: list[str], start: int, end: int) -> int:
    return sum(1 for ln in lines[start - 1:end] if ln.strip() and not _BOILERPLATE.match(ln))


def _node_name(node: Node) -> str:
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
    inner = node.child_by_field_name("definition")
    return (inner or node).type


@dataclass
class _Walk:
    ctx: ChunkContext
    lines: list[str]
    chunks: list[tuple[Chunk, str]] = field(default_factory=list)   # (chunk, parent qual)
    containers: list[tuple[int, int, str]] = field(default_factory=list)

    def emit_def(self, node: Node, parent: str, stack: list[tuple[Node, str]]) -> None:
        start = node.start_point[0] + 1
        # a node ending at column 0 of the next row does not really occupy that row
        end = node.end_point[0] + (0 if node.end_point[1] == 0 and node.end_point[0] > 0 else 1)
        end = min(max(end, start), len(self.lines))
        name = _node_name(node)
        qual = f"{parent}.{name}" if parent else name
        kind = _kind(node)
        chunk = self.ctx.build(self.lines, start, end, qual, kind)
        if kind in CONTAINER_TYPES and chunk.tokens > self.ctx.target_tokens:
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
                if kind in LEAF_TYPES or kind in CONTAINER_TYPES:
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
        # import/using blocks rank high on file-name similarity but carry no information
        if _meaningful(walk.lines, start, end) < MIN_GAP_NON_BLANK:
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
