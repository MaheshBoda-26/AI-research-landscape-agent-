"""Pipeline orchestration as an async generator of stage events.

One implementation drives both the SSE endpoint and the CLI. The generator yields
a ``StageEvent`` per transition; the caller decides whether to serialise those as
server-sent events or print them.

Four details are load-bearing:

**Blocking work is pushed to threads.** ``arxiv.Client`` wraps
``requests.Session``, cross-encoder inference is CPU-bound, and UMAP is
single-threaded numba. Called directly from the event loop, any of them stalls
the stream, and the symptom is a UI where all five stages appear at once at the
end instead of completing one by one.

**Connections do not cross threads.** A sqlite connection is opened and used
within a single thread. Workers open their own session inside the worker
function.

**Workers write to shared state rather than returning values.** An async
generator cannot ``return`` a value, so every worker assigns its result onto
``PipelineRun.state``. Because the generator awaits each future before reading
that state, there is no race.

**Intra-stage progress is real.** Extraction over 60 papers takes minutes, so a
progress callback pushes counts back to the event loop through a
``call_soon_threadsafe`` queue, and the generator yields them as the work runs.

Note on identifiers: ``state.local_labels`` maps ``paper_id`` to HDBSCAN's
**local** label (``-1`` for noise, ``0, 1, ...`` for clusters), not to the
``clusters`` table row id. Synthesis depends on that distinction to avoid naming
the unclustered bucket as if it were a research area.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from config import Settings
from llm.protocol import JSONCompleter
from models import (
    STAGE_ORDER,
    UNCLUSTERED_LABEL,
    Paper,
    PaperExtraction,
    RankedPaper,
    StageEvent,
    StageName,
    StageProgress,
)
from pipeline import cluster as cluster_mod
from pipeline import embed as embed_mod
from pipeline import extract as extract_mod
from pipeline import rerank as rerank_mod
from pipeline import retrieve as retrieve_mod
from pipeline import synthesize as synth_mod
from store import session as db_session

logger = logging.getLogger(__name__)

#: How often the generator checks for progress from a worker thread.
PROGRESS_POLL_SECONDS = 0.25

STAGE_LABELS: dict[StageName, str] = {
    "retrieval": "Searching arXiv",
    "rerank": "Ranking by relevance",
    "extraction": "Reading each paper",
    "layout": "Mapping the field",
    "synthesis": "Finding tensions and open problems",
}

ProgressCallback = Callable[[int, int], None]


@dataclass
class PipelineState:
    """Everything the stages hand to each other."""

    topic: str
    run_id: str
    landscape_id: int | None = None
    papers: list[Paper] = field(default_factory=list)
    ranked: list[RankedPaper] = field(default_factory=list)
    extractions: dict[str, PaperExtraction] = field(default_factory=dict)
    #: paper_id -> HDBSCAN local label (-1 means unclustered).
    local_labels: dict[str, int] = field(default_factory=dict)
    new_paper_count: int = 0

    @property
    def paper_ids(self) -> list[str]:
        return [paper.paper_id for paper in self.papers]

    @property
    def cluster_count(self) -> int:
        return len({label for label in self.local_labels.values() if label != UNCLUSTERED_LABEL})

    @property
    def unclustered_count(self) -> int:
        return sum(1 for label in self.local_labels.values() if label == UNCLUSTERED_LABEL)


class PipelineRun:
    """Runs one landscape build, yielding an event per stage transition."""

    def __init__(
        self,
        topic: str,
        settings: Settings,
        *,
        completer: JSONCompleter | None = None,
        expand_landscape_id: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.completer = completer
        self.expand_landscape_id = expand_landscape_id
        self.state = PipelineState(topic=topic, run_id=run_id or f"run_{uuid4().hex[:12]}")

    # -- event plumbing ----------------------------------------------------- #

    def _event(
        self,
        stage: StageName,
        status: str,
        message: str = "",
        *,
        progress: StageProgress | None = None,
        payload: dict[str, Any] | None = None,
    ) -> StageEvent:
        return StageEvent(
            run_id=self.state.run_id,
            landscape_id=self.state.landscape_id,
            stage=stage,
            status=status,  # type: ignore[arg-type]
            message=message or STAGE_LABELS.get(stage, ""),
            progress=progress or StageProgress(),
            payload=payload or {},
        )

    async def _run_blocking(
        self,
        stage: StageName,
        message: str,
        work: Callable[[ProgressCallback], None],
    ) -> AsyncIterator[StageEvent]:
        """Run blocking ``work`` in a thread, yielding live progress events.

        Yields nothing itself on success; the caller emits the stage's ``done``
        event. Any exception from ``work`` propagates to ``run``.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue()

        def report(current: int, total: int) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (current, total))

        future = loop.run_in_executor(None, work, report)
        try:
            while not future.done():
                try:
                    current, total = await asyncio.wait_for(
                        queue.get(), timeout=PROGRESS_POLL_SECONDS
                    )
                except TimeoutError:
                    continue
                yield self._event(
                    stage,
                    "running",
                    message,
                    progress=StageProgress(current=current, total=total),
                )
                # Let the transport flush between ticks.
                await asyncio.sleep(0)

            while not queue.empty():
                current, total = queue.get_nowait()
                yield self._event(
                    stage,
                    "running",
                    message,
                    progress=StageProgress(current=current, total=total),
                )

            await future  # re-raise anything the worker threw
        except BaseException:
            future.cancel()
            raise

    # -- stage 1: retrieval ------------------------------------------------- #

    async def _stage_retrieval(self) -> AsyncIterator[StageEvent]:
        yield self._event("retrieval", "running", f"Searching arXiv for {self.state.topic!r}")

        # ``completer`` is keyword-only on fetch_candidates; passing it
        # positionally would land on ``client`` and inject an LLM as an arXiv
        # client. Use a closure rather than to_thread's positional forwarding.
        papers = await asyncio.to_thread(
            lambda: retrieve_mod.fetch_candidates(
                self.state.topic, self.settings, completer=self.completer
            )
        )
        if not papers:
            raise retrieve_mod.RetrievalError(
                f"arXiv returned no papers for {self.state.topic!r}. Try a broader topic."
            )

        default_title = self.state.topic

        def persist() -> tuple[int, int]:
            with db_session(self.settings) as conn:
                topic_id = _upsert_topic(conn, self.state.topic)
                if self.expand_landscape_id is None:
                    landscape_id = _insert_landscape(
                        conn, topic_id=topic_id, title=default_title, params=_params(self.settings)
                    )
                else:
                    landscape_id = self.expand_landscape_id
                    _bump_generation(conn, landscape_id)
                    _update_landscape(conn, landscape_id, status="running")
                added = _upsert_papers(conn, papers)
                return landscape_id, added

        landscape_id, added = await asyncio.to_thread(persist)
        self.state.landscape_id = landscape_id
        self.state.new_paper_count = added

        if self.expand_landscape_id is not None:
            # Re-rank the whole corpus, not just the newcomers, so the ordering
            # of an expanded landscape stays globally correct.
            existing = await asyncio.to_thread(self._load_existing_papers)
            merged: dict[str, Paper] = {paper.paper_id: paper for paper in existing}
            for paper in papers:
                merged.setdefault(paper.paper_id, paper)
            papers = list(merged.values())
            logger.info(
                "Expanding landscape %s: %d existing + %d new = %d papers",
                landscape_id,
                len(existing),
                len(merged) - len(existing),
                len(merged),
            )

        self.state.papers = papers
        yield self._event(
            "retrieval",
            "done",
            f"Retrieved {len(papers)} papers",
            payload={"count": len(papers), "new": added},
        )

    def _load_existing_papers(self) -> list[Paper]:
        with db_session(self.settings) as conn:
            ids = _landscape_paper_ids(conn, self.state.landscape_id)
            papers = [_fetch_paper(conn, paper_id) for paper_id in ids]
        return [paper for paper in papers if paper is not None]

    # -- stage 2: rerank ---------------------------------------------------- #

    async def _stage_rerank(self) -> AsyncIterator[StageEvent]:
        total = len(self.state.papers)
        yield self._event(
            "rerank", "running", f"Scoring {total} candidates", progress=StageProgress(current=0, total=total)
        )

        settings = self.settings
        topic = self.state.topic
        papers = self.state.papers
        completer = self.completer

        def work(report: ProgressCallback) -> None:
            report(0, total)
            ranked = rerank_mod.rank_papers(topic, papers, settings, completer)
            self.state.ranked = rerank_mod.select_top(
                ranked, settings.rerank_final_count_capped
            )
            report(total, total)

        async for event in self._run_blocking("rerank", "Scoring candidates", work):
            yield event

        selected = self.state.ranked
        landscape_id = self.state.landscape_id

        def persist() -> None:
            with db_session(settings) as conn:
                for item in selected:
                    _link_paper(
                        conn,
                        landscape_id,
                        paper_id=item.paper.paper_id,
                        rank=item.rank,
                        relevance_score=item.relevance_score,
                        cross_encoder_logit=item.cross_encoder_logit,
                        rerank_source=item.rerank_source,
                        rationale=item.rationale,
                    )

        await asyncio.to_thread(persist)

        sources: dict[str, int] = {}
        for item in selected:
            sources[item.rerank_source] = sources.get(item.rerank_source, 0) + 1
        yield self._event(
            "rerank",
            "done",
            f"Selected the top {len(selected)} papers",
            payload={"selected": len(selected), "sources": sources},
        )

    # -- stage 3: extraction ------------------------------------------------ #

    async def _stage_extraction(self) -> AsyncIterator[StageEvent]:
        papers = [item.paper for item in self.state.ranked]
        total = len(papers)
        yield self._event(
            "extraction",
            "running",
            f"Reading {total} abstracts",
            progress=StageProgress(current=0, total=total),
        )

        settings = self.settings
        completer = self.completer

        def work(report: ProgressCallback) -> None:
            # Session opened inside the worker: one connection, one thread.
            with db_session(settings) as conn:
                self.state.extractions = extract_mod.extract_all(
                    papers, completer, settings, conn, on_progress=report
                )

        async for event in self._run_blocking("extraction", "Reading abstracts", work):
            yield event

        grounded = sum(
            1 for item in self.state.extractions.values() if extract_mod.has_grounded_content(item)
        )
        if self.completer is None:
            message = f"No LLM configured; skipped reading {total} papers"
        else:
            message = f"Extracted structure from {grounded} of {total} papers"
        yield self._event(
            "extraction",
            "done",
            message,
            payload={"extracted": grounded, "total": total, "llm": self.completer is not None},
        )

    # -- stage 4: layout ---------------------------------------------------- #

    async def _stage_layout(self) -> AsyncIterator[StageEvent]:
        papers = [item.paper for item in self.state.ranked]
        total = len(papers)
        yield self._event(
            "layout",
            "running",
            f"Projecting {total} papers",
            progress=StageProgress(current=0, total=total),
        )

        settings = self.settings
        landscape_id = self.state.landscape_id

        def work(report: ProgressCallback) -> None:
            from store import cluster_id_by_label, replace_clusters, update_layout

            report(0, total)
            with db_session(settings) as conn:
                vectors = embed_mod.embed_papers(papers, settings, conn)
            report(total // 3, total)

            matrix = embed_mod.embedding_matrix([p.paper_id for p in papers], vectors)
            result = cluster_mod.layout(matrix, settings)
            coords = cluster_mod.normalize_coords(result.coords)
            report((total * 2) // 3, total)

            centroids = cluster_mod.cluster_centroids(coords, result.labels)
            groups = cluster_mod.group_labels(result.labels)
            rows = [
                {
                    "local_label": label,
                    "paper_count": len(indices),
                    "x": centroids[label][0],
                    "y": centroids[label][1],
                    "color": cluster_mod.color_for_label(label),
                    "label": (
                        "Unclustered" if label == UNCLUSTERED_LABEL else f"Cluster {label}"
                    ),
                    "description": "",
                }
                for label, indices in sorted(groups.items())
            ]

            with db_session(settings) as conn:
                replace_clusters(conn, landscape_id, rows)
                mapping = cluster_id_by_label(conn, landscape_id)
                update_layout(
                    conn,
                    landscape_id,
                    {
                        papers[index].paper_id: (
                            float(coords[index][0]),
                            float(coords[index][1]),
                            mapping[int(result.labels[index])],
                        )
                        for index in range(len(papers))
                    },
                )
            # Local labels, not row ids: synthesis needs to recognise noise.
            self.state.local_labels = {
                papers[index].paper_id: int(result.labels[index])
                for index in range(len(papers))
            }
            report(total, total)

        async for event in self._run_blocking("layout", "Projecting papers", work):
            yield event

        yield self._event(
            "layout",
            "done",
            f"Found {self.state.cluster_count} cluster(s), "
            f"{self.state.unclustered_count} unclustered",
            payload={
                "clusters": self.state.cluster_count,
                "unclustered": self.state.unclustered_count,
            },
        )

    # -- stage 5: synthesis ------------------------------------------------- #

    async def _stage_synthesis(self) -> AsyncIterator[StageEvent]:
        yield self._event("synthesis", "running", "Building the research landscape")

        settings = self.settings
        completer = self.completer
        topic = self.state.topic
        landscape_id = self.state.landscape_id
        papers_by_id = {paper.paper_id: paper for paper in self.state.papers}
        members_by_label = synth_mod.group_by_cluster(self.state.paper_ids, self.state.local_labels)
        members = {
            label: [papers_by_id[pid] for pid in pids if pid in papers_by_id]
            for label, pids in members_by_label.items()
        }
        extraction_copy = dict(self.state.extractions)

        def work(report: ProgressCallback) -> None:
            from store import (
                replace_cluster_labels,
                replace_edges,
                replace_open_problems,
                replace_reading_path,
                replace_tensions,
            )

            real_clusters = max(1, len([k for k in members if k != UNCLUSTERED_LABEL]))
            report(0, real_clusters + 1)

            labels = synth_mod.label_clusters(
                topic, members, extraction_copy, completer, settings
            )
            report(real_clusters, real_clusters + 1)

            result = synth_mod.synthesize_landscape(
                topic,
                labels,
                members,
                self.state.papers,
                extraction_copy,
                completer,
                settings,
            )

            with db_session(settings) as conn:
                replace_cluster_labels(conn, landscape_id, result.clusters)
                replace_edges(conn, landscape_id, result.edges)
                replace_tensions(conn, landscape_id, result.tensions)
                replace_open_problems(conn, landscape_id, result.open_problems)
                replace_reading_path(conn, landscape_id, result.reading_path)
                _update_landscape(
                    conn,
                    landscape_id,
                    title=result.title or topic,
                    summary=result.summary,
                    status="ready",
                )
            report(real_clusters + 1, real_clusters + 1)

        async for event in self._run_blocking("synthesis", "Building the landscape", work):
            yield event

        summary = await asyncio.to_thread(self._read_landscape_summary)
        yield self._event("synthesis", "done", "Landscape ready", payload=summary)

    def _read_landscape_summary(self) -> dict[str, Any]:
        with db_session(self.settings) as conn:
            return {
                "edges": len(_fetch_edges(conn, self.state.landscape_id)),
                "tensions": len(_fetch_tensions(conn, self.state.landscape_id)),
                "open_problems": len(_fetch_open_problems(conn, self.state.landscape_id)),
                "clusters": self.state.cluster_count,
            }

    # -- driver ------------------------------------------------------------- #

    def _handler(self, stage: StageName) -> Callable[[], AsyncIterator[StageEvent]]:
        return {
            "retrieval": self._stage_retrieval,
            "rerank": self._stage_rerank,
            "extraction": self._stage_extraction,
            "layout": self._stage_layout,
            "synthesis": self._stage_synthesis,
        }[stage]

    async def run(self) -> AsyncIterator[StageEvent]:
        """Yield stage events until the landscape is ready or a stage fails."""
        for stage in STAGE_ORDER:
            try:
                async for event in self._handler(stage)():
                    yield event
            except retrieve_mod.RetrievalThrottled as exc:
                yield self._error(stage, exc, retryable=True)
                await asyncio.to_thread(self._mark_failed, exc)
                return
            except Exception as exc:  # noqa: BLE001 - report, never crash the stream
                logger.exception("Stage %s failed", stage)
                yield self._error(stage, exc, retryable=False)
                await asyncio.to_thread(self._mark_failed, exc)
                return

    def _error(self, stage: StageName, exc: Exception, *, retryable: bool) -> StageEvent:
        return self._event(
            stage,
            "error",
            str(exc) or exc.__class__.__name__,
            payload={"retryable": retryable, "error_type": exc.__class__.__name__},
        )

    def _mark_failed(self, exc: Exception) -> None:
        if self.state.landscape_id is None:
            return
        try:
            with db_session(self.settings) as conn:
                _update_landscape(conn, self.state.landscape_id, status="failed")
        except Exception:  # noqa: BLE001 - never mask the original failure
            logger.warning("Could not mark landscape %s failed", self.state.landscape_id)


# --------------------------------------------------------------------------- #
# Store imports, kept behind thin wrappers so the thread boundary is obvious
# and so this module does not import every store symbol at module scope.
# --------------------------------------------------------------------------- #


def _params(settings: Settings) -> dict[str, Any]:
    return {
        "prompt_version": settings.prompt_version,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "rerank_final_count": settings.rerank_final_count,
        "retrieval_max_results": settings.retrieval_max_results,
    }


def _upsert_topic(conn, topic: str) -> int:
    from store import upsert_topic

    return upsert_topic(conn, topic)


def _insert_landscape(conn, *, topic_id: int, title: str, params: dict[str, Any]) -> int:
    from store import insert_landscape

    return insert_landscape(conn, topic_id=topic_id, title=title, params=params)


def _bump_generation(conn, landscape_id: int) -> int:
    from store import bump_generation

    return bump_generation(conn, landscape_id)


def _landscape_paper_ids(conn, landscape_id: int) -> list[str]:
    from store import landscape_paper_ids

    return sorted(landscape_paper_ids(conn, landscape_id))


def _upsert_papers(conn, papers: list[Paper]) -> int:
    from store import upsert_papers

    return upsert_papers(conn, papers)


def _fetch_paper(conn, paper_id: str) -> Paper | None:
    from store import fetch_paper

    return fetch_paper(conn, paper_id)


def _link_paper(conn, landscape_id: int, **kwargs: Any) -> None:
    from store import link_paper

    link_paper(conn, landscape_id, **kwargs)


def _fetch_edges(conn, landscape_id: int) -> list[dict[str, Any]]:
    from store import fetch_edges

    return fetch_edges(conn, landscape_id)


def _fetch_tensions(conn, landscape_id: int) -> list[dict[str, Any]]:
    from store import fetch_tensions

    return fetch_tensions(conn, landscape_id)


def _fetch_open_problems(conn, landscape_id: int) -> list[dict[str, Any]]:
    from store import fetch_open_problems

    return fetch_open_problems(conn, landscape_id)


def _update_landscape(conn, landscape_id: int, **kwargs: Any) -> None:
    from store import update_landscape

    update_landscape(conn, landscape_id, **kwargs)


async def run_pipeline(
    topic: str,
    settings: Settings,
    *,
    completer: JSONCompleter | None = None,
    expand_landscape_id: int | None = None,
) -> AsyncIterator[StageEvent]:
    """Convenience wrapper: build a run and iterate it."""
    run = PipelineRun(
        topic, settings, completer=completer, expand_landscape_id=expand_landscape_id
    )
    async for event in run.run():
        yield event


__all__ = ["STAGE_LABELS", "PipelineRun", "PipelineState", "run_pipeline"]
