"""Translate a plain-English topic into an arXiv query string.

The arXiv API's ``search_query`` is not a natural-language interface: it wants
field-prefixed boolean expressions such as ``all:"retrieval-augmented
generation" OR all:grounding AND (cat:cs.CL OR cat:cs.LG)``. Keyword-matching a
raw topic string produces poor recall, so an LLM decomposes the topic first and
``build_query`` assembles the expression.

If no LLM is available the pipeline degrades to a quoted-phrase search of the
topic verbatim, which is a reasonable default and keeps the whole pipeline
runnable without a key.
"""

from __future__ import annotations

import re
from typing import ClassVar

from models import arXivQuery

# Categories that actually contain ML work. Used as a soft filter: applied only
# when the LLM returns them, never invented here.
ML_CATEGORIES: tuple[str, ...] = (
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.IR",
    "cs.LG",
    "cs.RO",
    "cs.NE",
    "stat.ML",
)

QUERY_SYSTEM_PROMPT = """You expand a plain-English machine-learning topic into arXiv search terms.

Return JSON with three arrays:
- "phrases": 2-4 exact multi-word phrases likely to appear verbatim in relevant
  paper titles or abstracts. Use the canonical technical phrasing, not the
  user's casual wording. Example: for "making AI remember things" use
  "retrieval-augmented generation" and "long-term memory".
- "keywords": 3-8 single technical terms that co-occur with the topic.
- "categories": 0-3 arXiv categories from this list, only when you are confident
  the topic lives there: cs.AI, cs.CL, cs.CV, cs.IR, cs.LG, cs.RO, cs.NE, stat.ML.

Rules:
- Never invent acronyms or benchmark names you are unsure about.
- Prefer fewer, more precise phrases over many vague ones.
- Do not include generic words like "learning", "model", or "deep" on their own.
- Respond with JSON only."""


def build_user_prompt(topic: str) -> str:
    return f"Topic: {topic}"


_WHITESPACE = re.compile(r"\s+")


def _clean_term(term: str) -> str:
    """Strip characters that would break the arXiv query grammar."""
    return _WHITESPACE.sub(" ", term).strip().strip('"').strip()


def heuristic_query(topic: str) -> arXivQuery:
    """No-LLM fallback: search for the topic as an exact phrase.

    Deliberately conservative. A quoted phrase over the whole topic has high
    precision, and if it returns nothing ``fetch_candidates`` retries with the
    same string unquoted.
    """
    cleaned = _clean_term(topic)
    return arXivQuery(phrases=[cleaned] if cleaned else [], keywords=[], categories=[])


def build_query(query: arXivQuery | str) -> str:
    """Assemble an arXiv ``search_query`` expression.

    Field prefixes are ``all:`` (any field) and ``cat:`` (category). Phrases are
    double-quoted, which arXiv treats as an exact-phrase match. The category
    clause is parenthesized and ANDed so that phrases OR together *within* the
    category restriction rather than letting one broad phrase escape it.
    """
    if isinstance(query, str):
        query = heuristic_query(query)

    phrases = [_clean_term(p) for p in query.phrases if _clean_term(p)]
    keywords = [_clean_term(k) for k in query.keywords if _clean_term(k)]
    categories = [_clean_term(c) for c in query.categories if _clean_term(c)]

    terms = [f'all:"{p}"' for p in phrases] + [f"all:{k}" for k in keywords]
    if not terms:
        return ""

    expression = " OR ".join(terms)
    if categories:
        category_clause = " OR ".join(f"cat:{c}" for c in categories)
        expression = f"({expression}) AND ({category_clause})"
    return expression


def plain_query(topic: str) -> str:
    """The unquoted retry used when the assembled query finds nothing."""
    cleaned = _clean_term(topic)
    return f"all:{cleaned}" if cleaned else ""


class QueryBuilderVersions:
    """Marker so the query prompt participates in prompt versioning."""

    CURRENT: ClassVar[str] = "query_v1"
