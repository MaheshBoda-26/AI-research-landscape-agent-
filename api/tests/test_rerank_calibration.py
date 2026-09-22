"""Tests for reranking: calibration, blending, and the degradation ladder.

No model is ever loaded. The cross-encoder is injected as a fake through the
``encoder=`` seam and the judge through the ``completer=`` seam, so the whole
stage is exercised offline and in milliseconds.

The calibration tests are the important ones. They encode the rule that scores
are absolute: a paper's score must not depend on what else happened to be in the
batch, because a batch-relative score silently makes every retrieved set look
equally good.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from config import Settings
from conftest import make_paper
from models import JudgeBatch, JudgeScore
from pipeline.rerank import (
    DEGRADED_SCORE_CAP,
    EncoderScores,
    calibrate,
    judge_papers,
    parse_judge_scores,
    rank_papers,
    relative_scores,
)


class FakeEncoder:
    """Returns preset calibrated scores (and optionally raw logits), or raises."""

    def __init__(
        self,
        scores: list[float] | None = None,
        logits: list[float] | None = None,
        raises: Exception | None = None,
    ):
        self._scores = scores
        self._logits = logits
        self._raises = raises
        self.calls: list[list[str]] = []

    def score(self, query: str, texts) -> EncoderScores:
        self.calls.append(list(texts))
        if self._raises is not None:
            raise self._raises
        calibrated = self._scores if self._scores is not None else [5.0] * len(texts)
        # Default raw mirrors the calibrated values: any monotonic stand-in is
        # sufficient because relative_scores only uses ordering.
        raw = self._logits if self._logits is not None else list(calibrated)
        return EncoderScores(raw=list(raw), calibrated=list(calibrated))


class FakeJudge:
    """Returns a JudgeBatch built by a callback, or raises."""

    def __init__(self, callback=None, raises: Exception | None = None):
        self._callback = callback
        self._raises = raises
        self.calls = 0
        self.last_user_prompt = ""

    @property
    def model_name(self) -> str:
        return "fake-judge"

    def complete_json(self, *, system, user, schema, max_repairs=None, temperature=0.0):
        self.calls += 1
        self.last_user_prompt = user
        if self._raises is not None:
            raise self._raises
        if self._callback is None:
            return JudgeBatch(scores=[])
        return self._callback(user)


def papers(n: int) -> list:
    return [make_paper(paper_id=f"p{i:03d}", title=f"Title {i}", abstract=f"Abstract {i}") for i in range(n)]


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_sigmoid_midpoint_is_neutral():
    assert calibrate([0.0])[0] == pytest.approx(5.0)


def test_large_positive_logit_saturates_high():
    assert calibrate([10.0])[0] > 9.9


def test_large_negative_logit_saturates_low():
    assert calibrate([-10.0])[0] < 0.1


def test_calibrate_is_monotonic():
    scores = calibrate([-8.0, -2.0, 0.0, 2.0, 8.0])
    assert scores == sorted(scores)


def test_calibrate_handles_extreme_logits_without_overflow():
    """Naive 1/(1+exp(-x)) overflows for very negative x."""
    assert calibrate([-1000.0])[0] == pytest.approx(0.0)
    assert calibrate([1000.0])[0] == pytest.approx(10.0)


def test_scores_are_absolute_and_not_batch_normalized():
    """A lone irrelevant paper must NOT be rescaled up to 10.

    This is the anti-min-max regression test: with batch normalization a single
    candidate always scores 10, however bad it is.
    """
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder(scores=[0.0001])
    ranked = rank_papers("topic", [make_paper()], settings, encoder=encoder)
    assert ranked[0].relevance_score < 0.01


def test_an_irrelevant_paper_stays_low_alongside_a_relevant_one():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder(scores=[9.5, 0.3])
    ranked = rank_papers("topic", papers(2), settings, encoder=encoder)
    bottom = ranked[-1]
    assert bottom.relevance_score == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# Degradation ladder
# --------------------------------------------------------------------------- #


def test_cross_encoder_failure_caps_at_neutral_and_keeps_input_order():
    """Rank order is real information; absolute relevance is not. Cap at neutral."""
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder(raises=RuntimeError("model missing"))
    ranked = rank_papers("topic", papers(3), settings, encoder=encoder)

    assert [r.paper.paper_id for r in ranked] == ["p000", "p001", "p002"]
    assert all(r.relevance_score == DEGRADED_SCORE_CAP for r in ranked)
    assert all(r.rerank_source == "fusion-fallback" for r in ranked)
    assert all(r.cross_encoder_score is None for r in ranked)


def test_mismatched_score_count_is_treated_as_failure():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder(scores=[9.0, 9.0])  # two scores for three papers
    ranked = rank_papers("topic", papers(3), settings, encoder=encoder)
    assert all(r.rerank_source == "fusion-fallback" for r in ranked)


def test_without_a_judge_scores_come_from_the_cross_encoder_alone():
    settings = replace(Settings.from_env(), rerank_seed_count=5)
    ranked = rank_papers("topic", papers(3), settings, None, encoder=FakeEncoder([7.0, 2.0, 5.0]))
    assert {r.rerank_source for r in ranked} == {"cross-encoder"}
    assert all(r.llm_score is None for r in ranked)


def test_judge_failure_still_yields_cross_encoder_results():
    settings = replace(Settings.from_env(), rerank_seed_count=5)
    judge = FakeJudge(raises=RuntimeError("judge exploded"))
    ranked = rank_papers("topic", papers(3), settings, judge, encoder=FakeEncoder([7.0, 2.0, 5.0]))
    assert len(ranked) == 3
    assert {r.rerank_source for r in ranked} == {"cross-encoder"}


def test_no_candidates_returns_empty():
    settings = Settings.from_env()
    assert rank_papers("topic", [], settings, None, encoder=FakeEncoder([])) == []


# --------------------------------------------------------------------------- #
# Saturation, measured on real data
# --------------------------------------------------------------------------- #


def test_signature_saturation_is_the_expected_failure_mode():
    """Real logits for a retrieved candidate set are all comfortably positive.

    Measured on 200 arXiv papers for "retrieval-augmented generation": raw
    logits spanned 5.31-9.71 (IQR 1.0), but sigmoid logits spanned only
    9.95-9.9994 and 156 of 200 papers rounded to >= 9.99. The ordering signal
    survives; the magnitude signal does not. This test pins that behaviour so a
    future change to the calibration cannot silently alter it.
    """
    logits = [5.31, 7.05, 7.59, 8.05, 9.71]
    calibrated = calibrate(logits)

    # Magnitude collapses: the whole spread is under a tenth of a point.
    assert max(calibrated) - min(calibrated) < 0.1
    # Ordering survives: the transform is monotonic.
    assert calibrated == sorted(calibrated)
    # And the raw logits are what still distinguish the candidates.
    assert max(logits) - min(logits) > 4.0


def test_saturated_scores_still_rank_correctly_but_relative_scores_separate_them():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    logits = [5.3, 7.1, 9.7]
    encoder = FakeEncoder(scores=calibrate(logits), logits=logits)
    ranked = rank_papers("topic", papers(3), settings, None, encoder=encoder)

    # Best paper first even though the calibrated scores are near-identical.
    assert [r.paper.paper_id for r in ranked] == ["p002", "p001", "p000"]
    assert ranked[0].relevance_score > 9.9 and ranked[-1].relevance_score > 9.9
    # The relative view restores separation for display.
    assert ranked[0].relative_score == pytest.approx(10.0)
    assert ranked[-1].relative_score == pytest.approx(0.0)
    assert ranked[0].cross_encoder_logit == pytest.approx(9.7)


def test_relative_scores_span_the_full_range():
    assert relative_scores([1.0, 2.0, 3.0]) == [0.0, 5.0, 10.0]


def test_relative_scores_of_a_single_candidate_is_neutral():
    assert relative_scores([42.0]) == [5.0]


def test_relative_scores_of_nothing_is_empty():
    assert relative_scores([]) == []


def test_relative_scores_share_average_rank_on_ties():
    # Two tied at the bottom share rank 0.5 -> 10 * 0.5 / 2 = 2.5
    assert relative_scores([1.0, 1.0, 3.0]) == [2.5, 2.5, 10.0]


def test_relative_scores_are_batch_relative_by_construction():
    """The same paper gets a different relative score in a different landscape.

    This is precisely why it must not be used for thresholds; relevance_score is
    the field that stays stable across candidate sets.
    """
    # The same value 5.0 is the worst candidate in one set and the best in another.
    assert relative_scores([5.0, 9.0])[0] == 0.0
    assert relative_scores([1.0, 5.0])[1] == 10.0


def test_fusion_fallback_carries_no_relative_or_logit_values():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder(raises=RuntimeError("no model"))
    ranked = rank_papers("topic", papers(2), settings, None, encoder=encoder)
    assert all(r.cross_encoder_logit is None and r.relative_score is None for r in ranked)


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #


def test_blend_uses_the_configured_weights():
    settings = replace(
        Settings.from_env(),
        rerank_seed_count=5,
        rerank_blend_ce=0.6,
        rerank_blend_llm=0.4,
    )
    judge = FakeJudge(callback=lambda _: JudgeBatch(scores=[JudgeScore(id=1, score=4.0, reason="meh")]))
    ranked = rank_papers("topic", papers(1), settings, judge, encoder=FakeEncoder([8.0]))

    assert ranked[0].rerank_source == "blend"
    assert ranked[0].cross_encoder_score == pytest.approx(8.0)
    assert ranked[0].llm_score == pytest.approx(4.0)
    assert ranked[0].relevance_score == pytest.approx(6.4)
    assert ranked[0].rationale == "meh"


def test_blending_can_reorder_the_cross_encoder_ranking():
    """The whole point of the judge: keyword-strong but off-topic papers drop."""
    settings = replace(Settings.from_env(), rerank_seed_count=5, rerank_blend_ce=0.5, rerank_blend_llm=0.5)
    # Cross-encoder likes p000 best; the judge strongly disagrees.
    encoder = FakeEncoder([9.0, 8.0])

    def callback(_):
        return JudgeBatch(
            scores=[
                JudgeScore(id=1, score=1.0),  # p000
                JudgeScore(id=2, score=9.0),  # p001
            ]
        )

    ranked = rank_papers("topic", papers(2), settings, FakeJudge(callback), encoder=encoder)
    assert ranked[0].paper.paper_id == "p001"
    assert ranked[0].rank == 1 and ranked[1].rank == 2


def test_judge_only_scores_the_seed_slice():
    """Beyond the seed count the cross-encoder ordering is reliable enough."""
    settings = replace(Settings.from_env(), rerank_seed_count=2, rerank_judge_batch_size=10)
    encoder = FakeEncoder([9.0, 8.0, 7.0, 6.0, 5.0])
    judged_ids: list[int] = []

    def callback(_user):
        # Only ever two papers should reach the judge.
        return JudgeBatch(scores=[JudgeScore(id=1, score=5.0), JudgeScore(id=2, score=5.0)])

    judge = FakeJudge(callback)
    ranked = rank_papers("topic", papers(5), settings, judge, encoder=encoder)

    assert judge.calls == 1
    blended = [r for r in ranked if r.rerank_source == "blend"]
    assert len(blended) == 2
    # The two highest cross-encoder scores are the ones that got judged.
    assert {r.paper.paper_id for r in blended} == {"p000", "p001"}
    assert judged_ids == []


def test_batching_sends_separate_calls():
    settings = replace(Settings.from_env(), rerank_seed_count=4, rerank_judge_batch_size=2)
    encoder = FakeEncoder([9.0, 8.0, 7.0, 6.0])
    judge = FakeJudge(
        callback=lambda _: JudgeBatch(
            scores=[JudgeScore(id=1, score=5.0), JudgeScore(id=2, score=5.0)]
        )
    )
    ranked = rank_papers("topic", papers(4), settings, judge, encoder=encoder)
    assert judge.calls == 2
    assert all(r.rerank_source == "blend" for r in ranked)


def test_ranks_are_dense_and_descending():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder([1.0, 9.0, 5.0])
    ranked = rank_papers("topic", papers(3), settings, None, encoder=encoder)
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert [r.relevance_score for r in ranked] == sorted(
        [r.relevance_score for r in ranked], reverse=True
    )


# --------------------------------------------------------------------------- #
# Judge response validation
# --------------------------------------------------------------------------- #


def _batch(*pairs) -> JudgeBatch:
    return JudgeBatch(scores=[JudgeScore(id=i, score=s) for i, s in pairs])


def test_parse_accepts_a_perfectly_aligned_response():
    parsed = parse_judge_scores(_batch((1, 8.0), (2, 2.0)), [1, 2])
    assert parsed == {1: (8.0, ""), 2: (2.0, "")}


def test_parse_rejects_a_missing_id():
    assert parse_judge_scores(_batch((1, 8.0)), [1, 2]) is None


def test_parse_rejects_an_extra_id():
    assert parse_judge_scores(_batch((1, 8.0), (2, 2.0), (3, 1.0)), [1, 2]) is None


def test_parse_rejects_a_duplicate_id():
    assert parse_judge_scores(_batch((1, 8.0), (1, 2.0)), [1, 2]) is None


def test_parse_rejects_a_none_response():
    assert parse_judge_scores(None, [1]) is None


def test_parse_rejects_an_empty_response():
    assert parse_judge_scores(JudgeBatch(scores=[]), [1]) is None


def test_parse_records_the_reason():
    batch = JudgeBatch(scores=[JudgeScore(id=1, score=7.0, reason="core topic paper")])
    assert parse_judge_scores(batch, [1]) == {1: (7.0, "core topic paper")}


def test_out_of_range_scores_are_rejected_by_the_schema():
    """Pydantic enforces 0-10 at the boundary, so a rogue score never arrives."""
    with pytest.raises(ValidationError):
        JudgeScore(id=1, score=42.0)


def test_misaligned_batch_is_discarded_not_partially_applied():
    settings = replace(Settings.from_env(), rerank_seed_count=2, rerank_judge_batch_size=10)
    encoder = FakeEncoder([9.0, 8.0])
    # Only one score for two papers: a shifted score would corrupt the ranking.
    judge = FakeJudge(callback=lambda _: _batch((1, 10.0)))
    ranked = rank_papers("topic", papers(2), settings, judge, encoder=encoder)
    assert {r.rerank_source for r in ranked} == {"cross-encoder"}


def test_judge_papers_without_a_completer_returns_nothing(settings: Settings):
    assert judge_papers("topic", papers(2), None, settings) == {}
