"""Keyword terms for the FTS5 index: identifier-aware, so exact names are findable.

Documents and queries both go through `terms()`: `getPlayerInventory`, `get_player_inventory`
and "player inventory" all share the terms `player` and `inventory`, and the whole identifier
is kept as its own term for exact matches.

Only ASCII words are indexed. The benchmark (evals/) showed CJK bigrams adding noise from
Chinese comments and docs, while the embedding model already handles Chinese queries well.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
# identifier-shaped: camelCase / PascalCase, snake_case, ALL_CAPS or letter+digit names
_IDENTIFIER = re.compile(
    r"[a-z][a-z0-9]*[A-Z]"            # camelCase
    r"|[A-Z][a-z0-9]+[A-Z]"           # PascalCase with 2+ humps
    r"|[A-Z]{2,}[a-z]"                # acronym first: HTTPServer, XMLParser, IOError
    r"|[A-Za-z0-9]_[A-Za-z0-9]"       # snake_case / ALL_CAPS_WITH_UNDERSCORE
    r"|\b[A-Z]{3,}\b"                 # ALLCAPS word (SQL keywords, constants)
    r"|[A-Za-z]{2,}[0-9]")            # letters then digits: utf8, md5sum, Base64
# (digit-first words such as "3rd" / "10px" are not identifiers and `terms()` skips them)
MIN_LEN = 2
MAX_QUERY_TERMS = 32
QUERY_STOPWORDS = frozenset({   # filler words in questions; no signal for code search
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "code", "do", "does", "for", "from",
    "function", "how", "if", "in", "into", "is", "it", "its", "method", "of", "on", "or", "so",
    "that", "the", "their", "then", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with",
})


def _identifier_terms(word: str) -> list[str]:
    parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
    whole = word.lower().strip("_")
    out = [whole] if len(whole) >= MIN_LEN else []
    if len(parts) > 1:
        out.extend(p for p in parts if len(p) >= MIN_LEN)
    return out


def terms(text: str) -> list[str]:
    return [t for word in _WORD.findall(text) for t in _identifier_terms(word)]


def document(text: str) -> str:
    """Space-joined terms, ready for an FTS5 column (unicode61 tokenizer)."""
    return " ".join(terms(text))


def has_identifier(query: str) -> bool:
    """True when the query names something code-like; only then is keyword search worth it."""
    return _IDENTIFIER.search(query) is not None


def match_expression(query: str) -> str | None:
    """FTS5 MATCH string OR-ing the query's distinct terms; None when nothing is searchable."""
    seen = dict.fromkeys(t for t in terms(query) if t not in QUERY_STOPWORDS)
    picked = list(seen)[:MAX_QUERY_TERMS]
    if not picked:
        return None
    return " OR ".join(f'"{t}"' for t in picked)
