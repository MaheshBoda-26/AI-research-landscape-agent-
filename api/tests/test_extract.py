"""Tests for structured extraction and evidence grounding.

The evidence validator is the only thing standing between this project and a map
that confidently describes papers incorrectly, so it is tested from both sides:
quotes that should be accepted (including typography-only differences that are
not the model's fault) and quotes that should be rejected (paraphrase, invention,
too short to mean anything).
"""

from __future__ import annotations

import pytest

import store
from config import Settings
from conftest import make_paper
from models import PaperExtraction
from pipeline.extract import (
    MIN_EVIDENCE_CHARS,
    enforce_evidence,
    extract_all,
    extract_one,
    is_verbatim,
    normalize_for_match,
    unextracted,
    ungrounded_fields,
)

ABSTRACT = (
    "Retrieval-augmented generation (RAG) improves factual grounding in large "
    "language models by conditioning generation on retrieved passages. We "
    "introduce GraphRAG, which builds an entity graph over a document corpus "
    "and retrieves subgraphs rather than flat passages. On the HotpotQA "
    "benchmark GraphRAG improves exact match from 41.2 to 48.7 while using 30% "
    "fewer retrieved tokens. Our main contribution is showing that structured "
    "retrieval indices outperform flat passage retrieval for multi-hop "
    "questions. The approach is not evaluated on open-ended generation."
)

PAPER = make_paper(paper_id="2406.12449", title="GraphRAG", abstract=ABSTRACT)

GROUNDED = {
    "problem": "Retrieval-augmented generation needs better grounding for multi-hop questions.",
    "method": "A graph index is built and subgraphs are retrieved instead of passages.",
    "results": "Exact match rises to 48.7 on HotpotQA.",
    "contribution": "Structured retrieval indices beat flat passage retrieval.",
    "limitations": "Open-ended generation is untested.",
    "datasets": ["HotpotQA"],
    "metrics": ["exact match"],
    "novelty": "substantial",
    "evidence": {
        "problem": "improves factual grounding in large language models by conditioning generation on retrieved passages",
        "method": "builds an entity graph over a document corpus and retrieves subgraphs rather than flat passages",
        "results": "improves exact match from 41.2 to 48.7 while using 30% fewer retrieved tokens",
        "contribution": "showing that structured retrieval indices outperform flat passage retrieval for multi-hop questions",
        "limitations": "not evaluated on open-ended generation",
    },
    "confidence": 0.9,
}


def grounded(**overrides) -> PaperExtraction:
    data = dict(GROUNDED)
    data.update(overrides)
    return PaperExtraction(**data)


class ScriptedCompleter:
    """Returns queued responses in order, recording the prompts it saw."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted-model"

    def complete_json(self, *, system, user, schema, max_repairs=None, temperature=0.0):
        self.prompts.append(user)
        if not self._responses:
            return None
        return self._responses.pop(0)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def test_normalization_collapses_whitespace_and_case():
    assert normalize_for_match("Hello   World\n\tAgain") == "hello world again"


def test_normalization_folds_unicode_typography_not_wording():
    assert normalize_for_match("a\u2014b") == "a-b"  # em dash
    assert normalize_for_match("a\u2013b") == "a-b"  # en dash
    assert normalize_for_match("\u201cquoted\u201d") == '"quoted"'
    assert normalize_for_match("diff\u00e9rent") == "différent"


def test_normalization_does_not_change_words():
    assert normalize_for_match("retrieval") != normalize_for_match("retrievals")


# --------------------------------------------------------------------------- #
# is_verbatim
# --------------------------------------------------------------------------- #


def test_exact_substring_is_verbatim():
    assert is_verbatim("builds an entity graph over a document corpus", ABSTRACT)


def test_whitespace_and_case_differences_are_tolerated():
    assert is_verbatim("BUILDS   an entity graph\nover a document corpus", ABSTRACT)


def test_unicode_dash_difference_is_tolerated():
    text = "A multi\u2014hop benchmark"
    assert is_verbatim("A multi-hop benchmark", text)


def test_paraphrase_is_rejected():
    assert not is_verbatim("the authors construct a knowledge graph", ABSTRACT)


def test_invented_quote_is_rejected():
    assert not is_verbatim("the model was trained on 100 GPUs for 3 weeks", ABSTRACT)


def test_quote_shorter_than_the_threshold_is_rejected():
    # A short fragment is a substring by accident, not evidence.
    assert len("entity graph") < MIN_EVIDENCE_CHARS
    assert not is_verbatim("entity graph", ABSTRACT)


def test_missing_quote_is_rejected():
    assert not is_verbatim(None, ABSTRACT)
    assert not is_verbatim("", ABSTRACT)


def test_empty_abstract_rejects_everything():
    assert not is_verbatim("builds an entity graph over a document corpus", "")


# --------------------------------------------------------------------------- #
# ungrounded_fields
# --------------------------------------------------------------------------- #


def test_a_fully_grounded_extraction_has_no_violations():
    assert ungrounded_fields(grounded(), ABSTRACT) == []


def test_a_fabricated_quote_flags_its_field():
    bad = grounded(evidence={**GROUNDED["evidence"], "results": "tripled throughput"})
    assert ungrounded_fields(bad, ABSTRACT) == ["results"]


def test_a_populated_field_with_no_evidence_entry_is_flagged():
    evidence = {k: v for k, v in GROUNDED["evidence"].items() if k != "method"}
    assert ungrounded_fields(grounded(evidence=evidence), ABSTRACT) == ["method"]


def test_null_fields_are_not_flagged():
    """Null is the correct answer for an unstated field, not a violation."""
    partial = PaperExtraction(
        problem="X", evidence={"problem": GROUNDED["evidence"]["problem"]}, novelty="unclear"
    )
    assert partial.method is None
    assert ungrounded_fields(partial, ABSTRACT) == []


def test_multiple_violations_are_all_reported():
    bad = grounded(
        evidence={
            "problem": GROUNDED["evidence"]["problem"],
            "method": "invented method text that is not present",
            "results": "invented results text that is not present",
        }
    )
    # contribution and limitations have no evidence entry at all; method and
    # results have entries whose quotes are invented.
    assert sorted(ungrounded_fields(bad, ABSTRACT)) == [
        "contribution",
        "limitations",
        "method",
        "results",
    ]


# --------------------------------------------------------------------------- #
# enforce_evidence
# --------------------------------------------------------------------------- #


def test_enforcement_leaves_a_clean_extraction_untouched():
    clean = grounded()
    assert enforce_evidence(clean, ABSTRACT) == clean


def test_enforcement_nulls_the_offending_field_only():
    bad = grounded(evidence={**GROUNDED["evidence"], "results": "fabricated result"})
    fixed = enforce_evidence(bad, ABSTRACT)
    assert fixed.results is None
    assert fixed.problem == GROUNDED["problem"]
    assert "results" not in fixed.evidence


def test_enforcement_penalises_confidence_proportionally():
    bad = grounded(evidence={**GROUNDED["evidence"], "results": "fabricated"})
    fixed = enforce_evidence(bad, ABSTRACT)
    # 4 of 5 prose fields survived, so confidence is scaled by 0.8.
    assert fixed.confidence == pytest.approx(0.9 * 0.8)


def test_enforcement_demotes_novelty_when_nothing_is_grounded():
    """A novelty judgement with no supporting text is an unsupported claim."""
    bad = grounded(
        evidence={"problem": "not in the abstract at all", "method": "also invented text here"},
    )
    fixed = enforce_evidence(bad, ABSTRACT)
    assert fixed.novelty == "unclear"
    assert fixed.confidence is None
    assert all(getattr(fixed, f) is None for f in ("problem", "method", "results", "contribution"))


def test_unextracted_has_no_prose_and_no_confidence():
    empty = unextracted()
    assert empty.problem is None and empty.confidence is None and empty.novelty == "unclear"


# --------------------------------------------------------------------------- #
# extract_one
# --------------------------------------------------------------------------- #


def test_extract_one_returns_grounded_fields(settings: Settings):
    completer = ScriptedCompleter([grounded()])
    result = extract_one(PAPER, completer, settings)
    assert result.problem == GROUNDED["problem"]
    assert result.datasets == ["HotpotQA"]
    assert len(completer.prompts) == 1


def test_extract_one_retries_once_when_evidence_is_fabricated(settings: Settings):
    """The correction pass is given the offending field names."""
    fabricated = grounded(evidence={**GROUNDED["evidence"], "results": "invented"})
    completer = ScriptedCompleter([fabricated, grounded()])
    result = extract_one(PAPER, completer, settings)

    assert result.results == GROUNDED["results"]
    assert len(completer.prompts) == 2
    assert "results" in completer.prompts[1]
    assert "verbatim" in completer.prompts[1].lower()


def test_extract_one_nulls_fields_when_the_retry_also_fails(settings: Settings):
    fabricated = grounded(evidence={**GROUNDED["evidence"], "results": "invented"})
    completer = ScriptedCompleter([fabricated, fabricated])
    result = extract_one(PAPER, completer, settings)
    assert result.results is None
    assert result.problem == GROUNDED["problem"]


def test_extract_one_does_not_retry_when_clean(settings: Settings):
    completer = ScriptedCompleter([grounded(), grounded()])
    extract_one(PAPER, completer, settings)
    assert len(completer.prompts) == 1


def test_extract_one_returns_a_null_record_on_llm_failure(settings: Settings):
    completer = ScriptedCompleter([])  # every call returns None
    result = extract_one(PAPER, completer, settings)
    assert result.problem is None and result.confidence is None


def test_extract_one_without_a_completer_is_unextracted(settings: Settings):
    result = extract_one(PAPER, None, settings)
    assert result.problem is None


def test_extract_one_gives_up_cleanly_if_the_retry_call_fails(settings: Settings):
    fabricated = grounded(evidence={**GROUNDED["evidence"], "results": "invented"})
    completer = ScriptedCompleter([fabricated])  # retry returns None
    result = extract_one(PAPER, completer, settings)
    assert result.problem is None


# --------------------------------------------------------------------------- #
# extract_all: caching, persistence, progress
# --------------------------------------------------------------------------- #


@pytest.fixture
def stored_papers(conn, settings: Settings):
    papers = [
        make_paper(paper_id="p1", abstract=ABSTRACT),
        make_paper(paper_id="p2", abstract=ABSTRACT),
    ]
    store.upsert_papers(conn, papers)
    return papers


def test_extract_all_runs_every_paper_once(settings: Settings, conn, stored_papers):
    completer = ScriptedCompleter([grounded(), grounded()])
    results = extract_all(stored_papers, completer, settings, conn)
    assert set(results) == {"p1", "p2"}
    assert len(completer.prompts) == 2


def test_second_run_is_fully_cached(settings: Settings, conn, stored_papers):
    """The whole point of keying the cache on prompt_version."""
    first = ScriptedCompleter([grounded(), grounded()])
    extract_all(stored_papers, first, settings, conn)
    assert len(first.prompts) == 2

    second = ScriptedCompleter([])  # would return None if called
    results = extract_all(stored_papers, second, settings, conn)
    assert len(second.prompts) == 0
    assert results["p1"].problem == GROUNDED["problem"]


def test_bumping_prompt_version_forces_re_extraction(settings: Settings, conn, stored_papers):
    from dataclasses import replace

    extract_all(stored_papers, ScriptedCompleter([grounded(), grounded()]), settings, conn)

    bumped = replace(settings, prompt_version="extract_v2")
    completer = ScriptedCompleter([grounded(), grounded()])
    extract_all(stored_papers, completer, bumped, conn)
    assert len(completer.prompts) == 2


def test_extractions_are_persisted_under_the_prompt_version(settings: Settings, conn, stored_papers):
    extract_all(stored_papers, ScriptedCompleter([grounded(), grounded()]), settings, conn)
    hit = store.fetch_extraction(conn, "p1", settings.prompt_version)
    assert hit is not None and hit.method == GROUNDED["method"]
    assert store.fetch_extraction(conn, "p1", "other_version") is None


def test_extract_all_without_a_completer_returns_null_records_but_caches_nothing(
    settings: Settings, conn, stored_papers
):
    """An absent LLM yields null extractions, not a failed run.

    Crucially they must NOT be cached: a null cached under this prompt_version
    would be a cache hit on the next run -- the one that finally has an API key
    -- and extraction would never happen.
    """
    results = extract_all(stored_papers, None, settings, conn)
    assert all(r.problem is None for r in results.values())
    assert store.fetch_extraction(conn, "p1", settings.prompt_version) is None


def test_a_failed_extraction_is_not_cached_so_it_retries(settings: Settings, conn, stored_papers):
    """A transient provider outage must not become permanent."""
    failing = ScriptedCompleter([None, None])
    extract_all(stored_papers, failing, settings, conn)
    assert store.fetch_extraction(conn, "p1", settings.prompt_version) is None

    working = ScriptedCompleter([grounded(), grounded()])
    extract_all(stored_papers, working, settings, conn)
    assert len(working.prompts) == 2
    assert store.fetch_extraction(conn, "p1", settings.prompt_version) is not None


def test_a_fully_ungrounded_extraction_is_not_cached(settings: Settings, conn, stored_papers):
    """Nothing usable was produced, so it does not count as extracted."""
    bogus = PaperExtraction(
        problem="invented", method="invented", evidence={"problem": "invented"}, confidence=0.9
    )
    extract_all(stored_papers, ScriptedCompleter([bogus, bogus, bogus, bogus]), settings, conn)
    assert store.fetch_extraction(conn, "p1", settings.prompt_version) is None


def test_progress_callback_walks_from_zero_to_total(settings: Settings, conn, stored_papers):
    seen: list[tuple[int, int]] = []
    extract_all(
        stored_papers,
        ScriptedCompleter([grounded(), grounded()]),
        settings,
        conn,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (2, 2)
    assert seen[0] == (0, 2)


def test_progress_counts_cached_papers_as_already_done(settings: Settings, conn, stored_papers):
    extract_all(stored_papers, ScriptedCompleter([grounded(), grounded()]), settings, conn)
    seen: list[tuple[int, int]] = []
    extract_all(
        stored_papers,
        ScriptedCompleter([]),
        settings,
        conn,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    # Everything was cached, so the callback never reports partial progress.
    assert seen == [(2, 2)]


def test_extract_all_of_nothing_is_empty(settings: Settings, conn):
    assert extract_all([], ScriptedCompleter([]), settings, conn) == {}


def test_extract_all_works_without_a_database(settings: Settings):
    results = extract_all([PAPER], ScriptedCompleter([grounded()]), settings, None)
    assert results["2406.12449"].problem == GROUNDED["problem"]
