"""Tests for arXiv query construction.

The query grammar is the difference between retrieving the right papers and
retrieving nothing, so these run without any network access.
"""

from __future__ import annotations

from config import Settings
from models import arXivQuery
from pipeline.retrieve import resolve_query
from prompts.query import ML_CATEGORIES, build_query, heuristic_query, plain_query


def test_phrases_are_quoted_and_ors_join_them():
    query = build_query(
        arXivQuery(
            phrases=["retrieval-augmented generation"],
            keywords=["RAG", "grounding"],
            categories=[],
        )
    )
    assert 'all:"retrieval-augmented generation"' in query
    assert "all:RAG" in query
    assert "all:grounding" in query
    assert query.count(" OR ") == 2
    # No categories means no parenthesized restriction clause.
    assert query.count("(") == 0


def test_categories_are_parenthesized_and_anded():
    query = build_query(
        arXivQuery(phrases=["diffusion policy"], keywords=[], categories=["cs.RO", "cs.LG"])
    )
    assert query.startswith('(all:"diffusion policy") AND (')
    assert "cat:cs.RO" in query and "cat:cs.LG" in query
    assert query.count("(") == query.count(")")
    assert query.count("(") == 2


def test_or_does_not_escape_the_category_restriction():
    """Without the outer parens, `A OR B AND C` would parse as `A OR (B AND C)`."""
    query = build_query(arXivQuery(phrases=["a"], keywords=["b"], categories=["cs.CL"]))
    assert query == '(all:"a" OR all:b) AND (cat:cs.CL)'


def test_empty_query_builds_an_empty_string():
    assert build_query(arXivQuery()) == ""


def test_quotes_and_whitespace_are_stripped_from_terms():
    query = build_query(arXivQuery(phrases=['  "spaced   out"  '], keywords=[]))
    assert query == 'all:"spaced out"'


def test_build_query_accepts_a_bare_string():
    assert build_query("graph neural networks") == 'all:"graph neural networks"'


def test_heuristic_query_quotes_the_whole_topic():
    result = heuristic_query("  retrieval-augmented   generation ")
    assert result.phrases == ["retrieval-augmented generation"]
    assert result.keywords == [] and result.categories == []


def test_heuristic_query_of_blank_topic_is_empty():
    assert heuristic_query("   ").phrases == []
    assert build_query(heuristic_query("   ")) == ""


def test_plain_query_is_unquoted_for_the_retry():
    assert plain_query("diffusion policy") == "all:diffusion policy"
    assert plain_query("  ") == ""


# --------------------------------------------------------------------------- #
# resolve_query — LLM expansion with degradation
# --------------------------------------------------------------------------- #


class FakeCompleter:
    def __init__(self, result=None, raises: Exception | None = None):
        self._result = result
        self._raises = raises
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake"

    def complete_json(self, *, system, user, schema, max_repairs=None, temperature=0.0):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._result


def test_resolve_query_without_a_completer_uses_the_raw_topic(settings: Settings):
    assert resolve_query("diffusion policy learning", settings, None) == heuristic_query(
        "diffusion policy learning"
    )


def test_resolve_query_uses_model_output(settings: Settings):
    completer = FakeCompleter(
        arXivQuery(phrases=["diffusion policy"], keywords=["imitation"], categories=["cs.RO"])
    )
    result = resolve_query("robot imitation", settings, completer)
    assert result.phrases == ["diffusion policy"]
    assert result.categories == ["cs.RO"]


def test_hallucinated_categories_are_dropped(settings: Settings):
    """A category that does not exist silently reduces recall to zero."""
    completer = FakeCompleter(
        arXivQuery(phrases=["x"], keywords=[], categories=["cs.LG", "cs.MADE_UP", "q-bio.QM"])
    )
    result = resolve_query("anything", settings, completer)
    assert result.categories == ["cs.LG"]
    assert all(c in ML_CATEGORIES for c in result.categories)


def test_resolve_query_survives_a_raising_model(settings: Settings):
    completer = FakeCompleter(raises=RuntimeError("upstream 500"))
    assert resolve_query("toxicology of LLMs", settings, completer) == heuristic_query(
        "toxicology of LLMs"
    )


def test_resolve_query_survives_an_empty_model_response(settings: Settings):
    completer = FakeCompleter(arXivQuery())
    assert resolve_query("sparse attention", settings, completer) == heuristic_query(
        "sparse attention"
    )


def test_resolve_query_survives_a_none_response(settings: Settings):
    completer = FakeCompleter(None)
    assert resolve_query("sparse attention", settings, completer) == heuristic_query(
        "sparse attention"
    )


def test_resolve_query_caps_phrase_and_keyword_counts(settings: Settings):
    completer = FakeCompleter(
        arXivQuery(phrases=[f"p{i}" for i in range(20)], keywords=[f"k{i}" for i in range(50)])
    )
    result = resolve_query("topic", settings, completer)
    assert len(result.phrases) == 4
    assert len(result.keywords) == 8
