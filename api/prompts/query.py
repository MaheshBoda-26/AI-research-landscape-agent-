"""Prompt templates for turning a plain-English topic into an arXiv query.

Kept separate from pipeline logic so query behavior can be tuned without
touching retrieval code, and so the prompt is reviewable in one place.
"""

from __future__ import annotations

QUERY_SYSTEM = """You convert a plain-English machine-learning topic into search
terms for the arXiv API.

Return 2-4 short exact phrases the literature actually uses, 2-5 single
keywords, and 1-3 arXiv category codes. Prefer canonical terminology over the
user's wording when they differ (a user asking about "vector databases for
LLMs" means phrases like "retrieval-augmented generation", not the literal
words).

Category codes are of the form `cs.LG`, `cs.CL`, `cs.AI`, `cs.CV`, `cs.RO`,
`stat.ML`, `eess.SY`, or a bare archive like `cs`. Omit categories entirely if
you are not confident which apply — a wrong category filter silently removes
the right papers.

Do not invent phrases that do not appear in real paper titles or abstracts."""

# The category whitelist the LLM is told to choose from, and that we validate
# against. A hallucinated category is silently catastrophic: it filters out
# every relevant paper and the run looks like a bad topic rather than a bug.
VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.NE", "cs.RO", "cs.SE",
        "cs.CR", "cs.DC", "cs.DS", "cs.HC", "cs.MA", "cs.MM", "cs.SD", "cs.SI",
        "stat.ML", "stat.AP", "stat.ME",
        "eess.AS", "eess.IV", "eess.SP", "eess.SY",
        "math.OC", "math.NA",
        "q-bio.QM", "q-bio.NC",
        "physics.comp-ph",
        "econ.EM",
        "cs", "stat", "math", "eess",
    }
)


def build_query_user(topic: str) -> str:
    return f"Topic: {topic}\n\nReturn the search terms as JSON."
