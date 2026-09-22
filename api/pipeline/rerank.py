"""Stage 2 — reranking.

Scores candidate papers by true semantic relevance to the topic, in two passes:

1. **Cross-encoder** over every candidate. Local, fast, free. Its raw logits are
   mapped to 0-10 through the model's own sigmoid, which for ms-marco
   cross-encoders is the trained relevance probability.
2. **LLM judge** over the top ``RERANK_SEED_COUNT`` only, in batches. Slower and
   costs tokens, so it is spent where it changes the outcome — a paper ranked
   120th by the cross-encoder is not going to be rescued.

The two signals are blended on the shared 0-10 scale, and every result records
which signals produced it (``rerank_source``) so the UI and the evaluation suite
can tell a confidently-blended score from a degraded one.

Calibration is the part that is easy to get wrong. Scores must be **absolute**:
``sigmoid(logit) * 10`` is query- and batch-independent, so a paper scoring 2.0
is genuinely weak evidence no matter what else was retrieved. Min-max normalizing
inside a batch would force the best candidate to 10 even when every candidate is
irrelevant, which both corrupts the UI and breaks any downstream threshold that
assumes the score means something. This is the same rationale documented in the
RAG-Pipeline project's ``cross_encoder_reranker.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from config import Settings
from llm.protocol import JSONCompleter
from models import JudgeBatch, Paper, RankedPaper
from prompts.rerank import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

logger = logging.getLogger(__name__)

#: Ceiling for a ranking that carries no absolute relevance signal.
#: Fusion/retrieval rank order is real information, but it says nothing about
#: *how* relevant a paper is, so a degraded score must not exceed neutral.
DEGRADED_SCORE_CAP = 5.0

#: Default cross-encoder. Trained on web search passages, which is a mismatch
#: for research prose but is fast, tiny (~90 MB), and adequate for the top-40
#: correction the judge then applies.
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class EncoderScores:
    """Both halves of a cross-encoder verdict.

    ``raw`` are unbounded logits and carry the *ordering* information.
    ``calibrated`` are the absolute 0-10 values and carry the *magnitude*
    information. Keeping both is what makes a saturated score range survivable.
    """

    raw: list[float]
    calibrated: list[float]


class CrossEncoderLike(Protocol):
    """What ``rank_papers`` needs from a scorer.

    Declared so tests can inject a fake and so no test ever downloads a model.
    Implementations own calibration and must return both raw and calibrated
    values.
    """

    def score(self, query: str, texts: Sequence[str]) -> EncoderScores: ...


# --------------------------------------------------------------------------- #
# Cross-encoder
# --------------------------------------------------------------------------- #


class CrossEncoderReranker:
    """Lazy wrapper around a sentence-transformers cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        # 512, not the 256 used elsewhere: abstracts run 1200-1900 characters
        # (~300-470 tokens) and a 256-token window truncates the tail, which is
        # exactly where the contribution and results usually sit.
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: CrossEncoderLike | None = None

    def _load_model(self) -> CrossEncoderLike:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
            logger.info("Loaded cross-encoder %s on %s", self.model_name, self.device)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> EncoderScores:
        """Return raw logits and calibrated 0-10 relevance, in input order."""
        if not texts:
            return EncoderScores(raw=[], calibrated=[])
        model = self._load_model()
        pairs = [(query, text) for text in texts]
        raw = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        logits = [float(score) for score in raw]
        return EncoderScores(raw=logits, calibrated=calibrate(logits))


def calibrate(raw_scores: Sequence[float]) -> list[float]:
    """Map raw logits to an absolute 0-10 scale via the model-native sigmoid.

    Deliberately not min-max: the result is query- and batch-independent, so a
    2.0 always means weak evidence. See the module docstring.

    Be aware of what this does to a *pre-filtered* candidate set. Measured on 200
    arXiv papers retrieved for "retrieval-augmented generation", raw logits
    spanned 5.31-9.71 (IQR 1.0) while these calibrated values spanned only
    9.95-9.9994 — 156 of 200 papers rounded to >= 9.99. The sigmoid preserves
    order but compresses magnitude, because everything retrieval returned really
    is about the topic. Use ``relative_scores`` when you need visual resolution.
    """
    return [round(10.0 * _sigmoid(float(score)), 4) for score in raw_scores]


def relative_scores(logits: Sequence[float]) -> list[float]:
    """Percentile-rank the logits onto 0-10 for display.

    RELATIVE BY CONSTRUCTION: a 10.0 means "best in this landscape", not
    "highly relevant". That is a different claim from ``relevance_score``, which
    is why it is stored in a different field and must never be used for a
    threshold or a refusal decision. It exists because otherwise a saturated
    score range makes an entire map look identical.

    Ties share the average rank.
    """
    count = len(logits)
    if count == 0:
        return []
    if count == 1:
        return [5.0]

    order = sorted(range(count), key=lambda index: logits[index])
    ranks = [0.0] * count
    start = 0
    while start < count:
        end = start
        while end + 1 < count and logits[order[end + 1]] == logits[order[start]]:
            end += 1
        average_rank = (start + end) / 2.0
        for position in range(start, end + 1):
            ranks[order[position]] = average_rank
        start = end + 1

    return [round(10.0 * rank / (count - 1), 4) for rank in ranks]


#: Process-wide default reranker, so the model loads once per server.
_reranker: CrossEncoderReranker | None = None


def get_reranker(settings: Settings) -> CrossEncoderReranker:
    """Return the shared reranker, rebuilding it if the configuration changed."""
    global _reranker
    if _reranker is None or (
        _reranker.model_name != settings.cross_encoder_model
        or _reranker.device != settings.cross_encoder_device
        or _reranker.max_length != settings.cross_encoder_max_length
    ):
        _reranker = CrossEncoderReranker(
            model_name=settings.cross_encoder_model,
            device=settings.cross_encoder_device,
            max_length=settings.cross_encoder_max_length,
            batch_size=settings.cross_encoder_batch_size,
        )
    return _reranker


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #


def parse_judge_scores(
    result: Any, expected_ids: Sequence[int]
) -> dict[int, tuple[float, str]] | None:
    """Validate a judge response against the ids that were actually sent.

    Returns ``None`` on any misalignment. A partial parse is worse than no parse:
    scores that are silently shifted by one position corrupt the ranking while
    looking perfectly healthy, so every deviation is rejected outright.
    """
    if result is None:
        return None
    scores = getattr(result, "scores", None)
    if not isinstance(scores, list) or not scores:
        return None

    out: dict[int, tuple[float, str]] = {}
    for entry in scores:
        entry_id = getattr(entry, "id", None)
        score = getattr(entry, "score", None)
        if entry_id is None or score is None:
            return None
        if not isinstance(entry_id, int):
            return None
        if entry_id in out:
            return None  # duplicate id
        if not isinstance(score, (int, float)):
            return None
        if not 0.0 <= float(score) <= 10.0:
            return None
        out[entry_id] = (float(score), getattr(entry, "reason", "") or "")

    if set(out) != set(expected_ids):
        return None
    return out


def judge_papers(
    topic: str,
    papers: list[Paper],
    completer: JSONCompleter | None,
    settings: Settings,
) -> dict[str, tuple[float, str]]:
    """Score papers with the LLM judge, in batches.

    Returns ``paper_id -> (score, reason)``. Papers the judge failed on are
    simply absent, which the caller treats as "no LLM signal for this paper"
    rather than a zero.

    Ids are 1-based *within each batch*, matching how the prompt numbers them.
    """
    if completer is None or not papers:
        return {}

    results: dict[str, tuple[float, str]] = {}
    batch_size = max(1, settings.rerank_judge_batch_size)

    for offset in range(0, len(papers), batch_size):
        batch = papers[offset : offset + batch_size]
        # Must match the numbering build_judge_user_prompt emits, which is 1-based
        # within this batch. Numbering globally here would misalign every batch
        # after the first, and parse_judge_scores would reject all of them.
        ids = list(range(1, len(batch) + 1))
        try:
            judged = completer.complete_json(
                system=JUDGE_SYSTEM_PROMPT,
                user=build_judge_user_prompt(topic, batch),
                schema=JudgeBatch,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - the judge is an optional signal
            logger.warning("Judge batch failed: %s", exc)
            continue

        parsed = parse_judge_scores(judged, ids)
        if parsed is None:
            logger.warning(
                "Discarding judge batch: response did not align with the %d ids sent", len(ids)
            )
            continue

        for local_index, paper in enumerate(batch):
            score, reason = parsed[ids[local_index]]
            results[paper.paper_id] = (score, reason)

    return results


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def rank_papers(
    topic: str,
    papers: list[Paper],
    settings: Settings,
    completer: JSONCompleter | None = None,
    *,
    encoder: CrossEncoderLike | None = None,
) -> list[RankedPaper]:
    """Rerank ``papers`` and return them best-first with ``rank`` populated.

    Degradation ladder, in order of preference:

    ``blend``            cross-encoder and judge both scored the paper
    ``cross-encoder``    only the cross-encoder scored it (judge off or failed)
    ``llm``              only the judge scored it
    ``fusion-fallback``  the cross-encoder failed; retrieval order kept, scores
                         capped at ``DEGRADED_SCORE_CAP`` because rank order
                         carries no absolute relevance signal
    """
    if not papers:
        return []

    scores: EncoderScores | None = None
    try:
        scorer = encoder if encoder is not None else get_reranker(settings)
        scores = scorer.score(topic, [paper.rerank_text for paper in papers])
    except Exception as exc:  # noqa: BLE001 - degrade to retrieval order
        logger.warning(
            "Cross-encoder failed (%s); falling back to retrieval order with capped scores", exc
        )

    counts_match = scores is not None and len(scores.calibrated) == len(papers) and len(scores.raw) == len(papers)
    if not counts_match:
        if scores is not None:
            logger.warning(
                "Cross-encoder returned %d calibrated / %d raw scores for %d papers; ignoring them",
                len(scores.calibrated),
                len(scores.raw),
                len(papers),
            )
        return [
            RankedPaper(
                paper=paper,
                relevance_score=DEGRADED_SCORE_CAP,
                cross_encoder_score=None,
                cross_encoder_logit=None,
                relative_score=None,
                llm_score=None,
                rerank_source="fusion-fallback",
                rank=index,
                rationale="Reranker unavailable; retrieval order preserved.",
            )
            for index, paper in enumerate(papers, start=1)
        ]

    assert scores is not None  # narrowed by counts_match
    ce_scores = scores.calibrated
    logits = scores.raw
    # Relative only; see relative_scores for why this is safe to display and
    # unsafe to threshold on.
    relative = relative_scores(logits)

    # The judge only sees the top slice. Beyond it, the cross-encoder ordering is
    # reliable enough that spending tokens would not change the outcome.
    seed_count = max(1, settings.rerank_seed_count)
    ranked_by_ce = sorted(
        zip(papers, ce_scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    judge_targets = [paper for paper, _ in ranked_by_ce[:seed_count]]
    judge_scores = judge_papers(topic, judge_targets, completer, settings)

    weight_ce = settings.rerank_blend_ce
    weight_llm = settings.rerank_blend_llm

    ranked: list[RankedPaper] = []
    for index, (paper, ce_score) in enumerate(zip(papers, ce_scores, strict=True)):
        judged = judge_scores.get(paper.paper_id)
        llm_score = judged[0] if judged else None
        reason = judged[1] if judged else ""

        if llm_score is not None:
            final = weight_ce * ce_score + weight_llm * llm_score
            source = "blend"
        else:
            final = ce_score
            source = "cross-encoder"

        ranked.append(
            RankedPaper(
                paper=paper,
                relevance_score=round(min(10.0, max(0.0, final)), 4),
                cross_encoder_score=ce_score,
                cross_encoder_logit=logits[index],
                relative_score=relative[index],
                llm_score=llm_score,
                rerank_source=source,  # type: ignore[arg-type]
                rationale=reason,
            )
        )

    ranked.sort(key=lambda item: (-item.relevance_score, item.paper.paper_id))
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked


def select_top(ranked: list[RankedPaper], limit: int) -> list[RankedPaper]:
    """Take the top ``limit`` papers, preserving rank."""
    return ranked[: max(0, limit)]


__all__ = [
    "DEFAULT_CROSS_ENCODER",
    "DEGRADED_SCORE_CAP",
    "CrossEncoderLike",
    "CrossEncoderReranker",
    "EncoderScores",
    "calibrate",
    "get_reranker",
    "judge_papers",
    "parse_judge_scores",
    "rank_papers",
    "relative_scores",
    "select_top",
]
