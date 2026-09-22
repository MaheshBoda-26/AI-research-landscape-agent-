"""Pydantic models: the data contract shared by every pipeline stage.

Two groups live here:

1. **Pipeline models** — what stages consume and produce (``Paper``, the LLM
   output schemas, ``StageEvent``).
2. **API models** — what the FastAPI layer returns to the frontend. These are
   the contract mirrored by hand in ``web/lib/types.ts``.

The LLM output schemas are deliberately narrow. Every field the LLM produces is
either a constrained enum, a grounded reference to a supplied paper ID, or a
verbatim quote from the source abstract, so that a hallucination is a
validation failure rather than a plausible-looking addition to the map.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Enums / literals
# --------------------------------------------------------------------------- #

StageName = Literal["retrieval", "rerank", "extraction", "synthesis", "layout"]
StageStatus = Literal["running", "done", "error"]
EdgeKind = Literal["extends", "contradicts", "applies", "shares_method"]
Novelty = Literal["incremental", "substantial", "unclear"]
LandscapeStatus = Literal["running", "ready", "failed"]

# Fixed execution order. The SSE stream and the UI timeline both assume this.
STAGE_ORDER: tuple[StageName, ...] = (
    "retrieval",
    "rerank",
    "extraction",
    "synthesis",
    "layout",
)

UNCLUSTERED_LABEL = -1
UNCLUSTERED_NAME = "Unclustered"


def utcnow() -> str:
    """ISO-8601 UTC timestamp, second precision, always suffixed ``Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Pipeline models
# --------------------------------------------------------------------------- #


class Paper(BaseModel):
    """A single arXiv result, normalized across library versions.

    ``paper_id`` is the **version-stripped** short id (``2107.05580``), because
    ``arxiv.Result.__eq__`` compares ``entry_id`` which *includes* the ``vN``
    suffix — so v1 and v3 of the same paper are distinct results and would both
    land in the map without stripping.
    """

    paper_id: str
    version: str = ""
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    updated: str = ""
    primary_category: str = ""
    categories: list[str] = Field(default_factory=list)
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    abs_url: str = ""
    pdf_url: str = ""

    @property
    def abs_link(self) -> str:
        """Canonical abstract page. arXiv asks that users be sent here."""
        return self.abs_url or f"https://arxiv.org/abs/{self.paper_id}"

    @property
    def rerank_text(self) -> str:
        """Text handed to the cross-encoder and the judge."""
        return f"{self.title}\n\n{self.abstract}"


class arXivQuery(BaseModel):
    """LLM-produced decomposition of a plain-English topic into arXiv syntax."""

    phrases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)

    @field_validator("phrases", "keywords", "categories", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, value: object) -> object:
        return [] if value is None else value


class RankedPaper(BaseModel):
    """A paper with its rerank verdict attached.

    Two relevance numbers are carried on purpose. ``relevance_score`` is the
    absolute, calibrated 0-10 value you can threshold on. ``cross_encoder_logit``
    is the raw model output, which is what actually resolves *ordering* among
    candidates: for a pre-filtered candidate set the calibrated score saturates
    near 10 for almost everything, while the logits still spread over a range of
    several points.
    """

    paper: Paper
    relevance_score: float = Field(ge=0.0, le=10.0)
    cross_encoder_score: float | None = None
    cross_encoder_logit: float | None = None
    relative_score: float | None = Field(default=None, ge=0.0, le=10.0)
    llm_score: float | None = None
    rerank_source: Literal["cross-encoder", "llm", "blend", "fusion-fallback"] = "cross-encoder"
    rank: int = 0
    rationale: str = ""


class PaperExtraction(BaseModel):
    """Structured reading of one abstract.

    ``evidence`` must contain a verbatim span of the abstract for each populated
    field. The validator in ``pipeline.extract`` checks substring containment, so
    a fabricated quote is rejected as ungrounded rather than stored.
    """

    problem: str | None = None
    method: str | None = None
    results: str | None = None
    contribution: str | None = None
    limitations: str | None = None
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    novelty: Novelty = "unclear"
    evidence: dict[str, str] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    # ClassVar, not a field: these are the keys ``evidence`` is expected to
    # cover, not data carried by an instance.
    EXTRACTED_FIELDS: ClassVar[tuple[str, ...]] = (
        "problem",
        "method",
        "results",
        "contribution",
        "limitations",
    )

    @field_validator("datasets", "metrics", "evidence", mode="before")
    @classmethod
    def _coerce_none(cls, value: object) -> object:
        return [] if value is None else value


class JudgeScore(BaseModel):
    """One paper's relevance verdict from the LLM judge."""

    id: int
    score: float = Field(ge=0.0, le=10.0)
    reason: str = ""


class JudgeBatch(BaseModel):
    """The judge's response envelope.

    An object containing an array rather than a bare array, so the response
    validates against a Pydantic model like every other LLM call in the project.
    """

    scores: list[JudgeScore] = Field(default_factory=list)


class ClusterLabel(BaseModel):
    """LLM naming of one computed cluster."""

    local_label: int
    label: str
    description: str = ""
    representative_paper_ids: list[str] = Field(default_factory=list)


class EdgeClaim(BaseModel):
    """One relationship between two papers, both of which must be supplied."""

    src_paper_id: str
    dst_paper_id: str
    kind: EdgeKind
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class Tension(BaseModel):
    """A genuine disagreement between two papers, not merely a difference."""

    statement: str
    paper_a_id: str
    paper_b_id: str


class OpenProblem(BaseModel):
    statement: str
    why_open: str = ""
    supporting_paper_ids: list[str] = Field(default_factory=list)


class ReadingStep(BaseModel):
    """One entry in the suggested reading order."""

    paper_id: str
    position: int
    why: str = ""


class LandscapeSynthesis(BaseModel):
    """The full synthesis output for one landscape."""

    title: str
    summary: str
    clusters: list[ClusterLabel] = Field(default_factory=list)
    edges: list[EdgeClaim] = Field(default_factory=list)
    tensions: list[Tension] = Field(default_factory=list)
    open_problems: list[OpenProblem] = Field(default_factory=list)
    reading_path: list[ReadingStep] = Field(default_factory=list)


class StageProgress(BaseModel):
    current: int = 0
    total: int = 0


class StageEvent(BaseModel):
    """One SSE frame. The frontend timeline consumes exactly this shape."""

    run_id: str
    landscape_id: int | None = None
    stage: StageName
    status: StageStatus
    message: str = ""
    progress: StageProgress = Field(default_factory=StageProgress)
    payload: dict = Field(default_factory=dict)
    ts: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# API models — mirrored by hand in web/lib/types.ts
# --------------------------------------------------------------------------- #


class ClusterOut(BaseModel):
    id: int
    label: str
    description: str = ""
    paper_count: int = 0
    x: float = 0.0
    y: float = 0.0
    color: str = "#94a3b8"
    is_unclustered: bool = False


class PaperInLandscape(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    primary_category: str = ""
    categories: list[str] = Field(default_factory=list)
    abs_url: str
    pdf_url: str = ""
    rank: int
    relevance_score: float
    cross_encoder_logit: float | None = None
    #: Percentile rank within this landscape, 0-10. RELATIVE, unlike
    #: ``relevance_score``, and only meaningful for display and sizing.
    relative_score: float | None = None
    rerank_source: str
    cluster_id: int | None = None
    x: float = 0.0
    y: float = 0.0
    is_seed: bool = False
    extraction: PaperExtraction | None = None


class EdgeOut(BaseModel):
    src_paper_id: str
    dst_paper_id: str
    kind: EdgeKind
    weight: float
    rationale: str = ""


class TensionOut(BaseModel):
    statement: str
    paper_a_id: str
    paper_b_id: str


class OpenProblemOut(BaseModel):
    statement: str
    why_open: str = ""
    supporting_paper_ids: list[str] = Field(default_factory=list)


class ReadingStepOut(BaseModel):
    paper_id: str
    position: int
    why: str = ""
    title: str = ""


class LandscapeSummary(BaseModel):
    id: int
    topic: str
    title: str
    summary: str = ""
    status: LandscapeStatus
    generation: int
    paper_count: int
    cluster_count: int
    created_at: str
    updated_at: str


class LandscapeDetail(BaseModel):
    id: int
    topic: str
    title: str
    summary: str
    status: LandscapeStatus
    generation: int
    created_at: str
    updated_at: str
    clusters: list[ClusterOut] = Field(default_factory=list)
    papers: list[PaperInLandscape] = Field(default_factory=list)
    edges: list[EdgeOut] = Field(default_factory=list)
    tensions: list[TensionOut] = Field(default_factory=list)
    open_problems: list[OpenProblemOut] = Field(default_factory=list)
    reading_path: list[ReadingStepOut] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    llm_configured: bool = False
    llm_provider: str = "nim"
    llm_model: str = ""
    prompt_version: str = ""


class TopicRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)

    @field_validator("topic")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 2:
            raise ValueError("topic must contain at least 2 non-space characters")
        return cleaned


class ExpandRequest(BaseModel):
    max_new_results: int = Field(default=100, ge=1, le=500)
