"""Query construction and query-expansion fallback tests."""

from __future__ import annotations

from models import arXivQuery
from pipeline.retrieve import build_query, resolve_query


class StubLLM:
    """Duck-typed stand-in for LLMClient.complete_json."""

    def __init__(self, result=None, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[dict] = []

    def complete_json(self, *, system: str, user: str, schema: type) -> object:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.raises is not None:
            raise self.raises
        return self.result


# --------------------------------------------------------------------------- #
# build_query
# --------------------------------------------------------------------------- #


def test_query_quotes_phrases_and_filters_categories():
    query = build_query(
        arXivQuery(
            phrases=["retrieval-augmented generation"],
            keywords=["RAG", "grounding"],
            categories=["cs.CL", "cs.LG"],
        )
    )
    assert 'all:"retrieval-augmented generation"' in query
    assert "all:RAG" in query and " OR " in query
    assert "cat:cs.CL" in query and "cat:cs.LG" in query
    assert query.count("(") == query.count(")")


def test_query_without_categories_has_no_grouping():
    query = build_query(arXivQuery(phrases=["diffusion policy"]))
    assert query == 'all:"diffusion policy"'
    assert "cat:" not in query


def test_query_collapses_internal_whitespace():
    query = build_query(arXivQuery(phrases=["  retrieval   augmented\n generation "]))
    assert query == 'all:"retrieval augmented generation"'


def test_empty_query_renders_empty_string():
    assert build_query(arXivQuery()) == ""


def test_blank_terms_are_skipped():
    query = build_query(arXivQuery(phrases=["  ", ""], keywords=["RAG"]))
    assert query == "all:RAG"


# --------------------------------------------------------------------------- #
# resolve_query
# --------------------------------------------------------------------------- #


def test_resolve_query_falls_back_to_literal_topic_without_llm():
    resolved = resolve_query("diffusion policy learning", llm=None)
    assert resolved.phrases == ["diffusion policy learning"]
    assert resolved.keywords == []


def test_resolve_query_uses_llm_output():
    llm = StubLLM(
        arXivQuery(
            phrases=["retrieval-augmented generation"],
            keywords=["RAG"],
            categories=["cs.CL"],
        )
    )
    resolved = resolve_query("RAG", llm=llm)
    assert resolved.phrases == ["retrieval-augmented generation"]
    assert resolved.categories == ["cs.CL"]
    assert llm.calls[0]["schema"] is arXivQuery


def test_resolve_query_falls_back_when_llm_raises():
    llm = StubLLM(raises=RuntimeError("provider down"))
    resolved = resolve_query("quantization for LLMs", llm=llm)
    assert resolved.phrases == ["quantization for LLMs"]


def test_resolve_query_falls_back_when_llm_returns_none():
    resolved = resolve_query("mixture of experts", llm=StubLLM(result=None))
    assert resolved.phrases == ["mixture of experts"]


def test_resolve_query_falls_back_when_llm_returns_empty_schema():
    resolved = resolve_query("sparse attention", llm=StubLLM(result=arXivQuery()))
    assert resolved.phrases == ["sparse attention"]


def test_resolve_query_drops_hallucinated_categories():
    """A wrong category filter silently removes the right papers, so it is dropped."""
    llm = StubLLM(
        arXivQuery(
            phrases=["vision transformers"],
            categories=["cs.LG", "cs.MADE_UP", "nonexistent.XX"],
        )
    )
    resolved = resolve_query("vision transformers", llm=llm)
    assert resolved.categories == ["cs.LG"]


def test_resolve_query_dedupes_terms_case_insensitively():
    llm = StubLLM(
        arXivQuery(phrases=["RAG", "rag", "RAG"], keywords=["Grounding", "grounding"])
    )
    resolved = resolve_query("rag", llm=llm)
    assert resolved.phrases == ["RAG"]
    assert resolved.keywords == ["Grounding"]


def test_resolve_query_caps_term_counts():
    llm = StubLLM(
        arXivQuery(
            phrases=[f"phrase {i}" for i in range(10)],
            keywords=[f"kw{i}" for i in range(10)],
            categories=["cs.CL", "cs.LG", "cs.AI", "cs.CV"],
        )
    )
    resolved = resolve_query("topic", llm=llm)
    assert len(resolved.phrases) == 4
    assert len(resolved.keywords) == 5
    assert len(resolved.categories) == 3
