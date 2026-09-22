"""Stage 4 — synthesis.

Turns 60 extractions plus a computed layout into a narrated landscape: named
clusters, typed relationships, real disagreements, open questions, and an entry
reading order.

Two passes, because one prompt asking for everything at once produces vague
output on both halves:

1. **Label each cluster** from its members' extracted problem/contribution.
   Cheap, parallel, and independently degradable — a cluster that fails to label
   still exists, just with a placeholder name.
2. **Synthesize the landscape** from the cluster structure plus the strongest
   papers' extractions.

The critical guard is ``validate_synthesis``: any edge, tension, open problem, or
reading step that references a paper id we did not supply is dropped. A model
that invents an arXiv id produces a graph edge pointing at nothing, which is both
unrenderable and a quiet lie about the literature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from config import Settings
from llm.protocol import JSONCompleter
from models import (
    UNCLUSTERED_LABEL,
    ClusterLabel,
    ClusterNaming,
    LandscapeSynthesis,
    Paper,
    PaperExtraction,
)
from prompts.cluster import CLUSTER_LABEL_SYSTEM_PROMPT, build_cluster_user_prompt
from prompts.synthesize import SYNTHESIS_SYSTEM_PROMPT, build_synthesis_user_prompt

logger = logging.getLogger(__name__)

#: Upper bounds, so a runaway response cannot produce an unreadable map.
MAX_EDGES = 60
MAX_TENSIONS = 8
MAX_OPEN_PROBLEMS = 10
MAX_READING_PATH = 12


def placeholder_label(local_label: int) -> ClusterLabel:
    """Fallback when a cluster cannot be named. Visible, not silent."""
    return ClusterLabel(
        local_label=local_label,
        label=f"Cluster {local_label}",
        description="This cluster could not be named.",
    )


def label_clusters(
    topic: str,
    members_by_label: dict[int, list[Paper]],
    extractions: dict[str, PaperExtraction],
    completer: JSONCompleter | None,
    settings: Settings,
) -> list[ClusterLabel]:
    """Name every real cluster. ``-1`` (noise) is never named as a topic."""
    real = {label: papers for label, papers in members_by_label.items() if label != UNCLUSTERED_LABEL}
    if not real:
        return []

    # Start from placeholders so a failed labelling call still yields a cluster.
    named: dict[int, ClusterLabel] = {
        label: placeholder_label(label) for label in sorted(real)
    }

    if completer is None:
        logger.warning("No LLM client; %d cluster(s) keep placeholder names", len(real))
        return [named[label] for label in sorted(named)]

    def work(item: tuple[int, list[Paper]]) -> tuple[int, ClusterLabel]:
        label, papers = item
        payload = [(paper, extractions.get(paper.paper_id)) for paper in papers]
        naming = completer.complete_json(
            system=CLUSTER_LABEL_SYSTEM_PROMPT,
            user=build_cluster_user_prompt(topic, label, payload),
            schema=ClusterNaming,
            temperature=0.0,
        )
        if naming is None or not naming.label.strip():
            return label, placeholder_label(label)
        valid_ids = {paper.paper_id for paper in papers}
        return label, ClusterLabel(
            local_label=label,
            label=" ".join(naming.label.split())[:80],
            description=" ".join(naming.description.split()),
            # Representative ids must come from this cluster's members.
            representative_paper_ids=[
                pid for pid in naming.representative_paper_ids if pid in valid_ids
            ][:4],
        )

    workers = max(1, min(settings.llm_concurrency, len(real)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cluster-label") as pool:
        for label, cluster in pool.map(work, sorted(real.items())):
            named[label] = cluster

    return [named[label] for label in sorted(named)]


def validate_synthesis(
    synthesis: LandscapeSynthesis, valid_ids: set[str]
) -> LandscapeSynthesis:
    """Drop every reference to a paper that was not supplied.

    Also removes self-loops, de-duplicates edges, and renumbers reading steps so
    the order stays dense after any drops.
    """
    dropped = 0

    edges = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in synthesis.edges:
        if edge.src_paper_id not in valid_ids or edge.dst_paper_id not in valid_ids:
            dropped += 1
            continue
        if edge.src_paper_id == edge.dst_paper_id:
            dropped += 1
            continue
        key = (edge.src_paper_id, edge.dst_paper_id, edge.kind)
        if key in seen_edges:
            dropped += 1
            continue
        seen_edges.add(key)
        edges.append(edge)

    reading_path = []
    kept_steps = [
        step for step in synthesis.reading_path if step.paper_id in valid_ids
    ]
    dropped += len(synthesis.reading_path) - len(kept_steps)
    # De-duplicate, then renumber densely from 1 in the model's intended order.
    ordered = sorted(kept_steps, key=lambda step: (step.position, step.paper_id))
    seen_papers: set[str] = set()
    for step in ordered:
        if step.paper_id in seen_papers:
            dropped += 1
            continue
        seen_papers.add(step.paper_id)
        step = step.model_copy(update={"position": len(reading_path) + 1})
        reading_path.append(step)

    tensions = []
    for tension in synthesis.tensions:
        if tension.paper_a_id not in valid_ids or tension.paper_b_id not in valid_ids:
            dropped += 1
            continue
        if tension.paper_a_id == tension.paper_b_id:
            dropped += 1
            continue
        tensions.append(tension)

    problems = []
    for problem in synthesis.open_problems:
        unknown = [pid for pid in problem.supporting_paper_ids if pid not in valid_ids]
        dropped += len(unknown)
        problems.append(
            problem.model_copy(
                update={
                    "supporting_paper_ids": [
                        pid for pid in problem.supporting_paper_ids if pid in valid_ids
                    ]
                }
            )
        )

    if dropped:
        logger.warning("Dropped %d unverifiable reference(s) from the synthesis", dropped)

    return synthesis.model_copy(
        update={
            "edges": edges[:MAX_EDGES],
            "tensions": tensions[:MAX_TENSIONS],
            "open_problems": problems[:MAX_OPEN_PROBLEMS],
            "reading_path": reading_path[:MAX_READING_PATH],
        }
    )


def fallback_synthesis(
    topic: str, clusters: Sequence[ClusterLabel], papers: Sequence[Paper]
) -> LandscapeSynthesis:
    """A minimal but honest landscape when synthesis is unavailable.

    Says plainly that synthesis did not run, rather than presenting computed
    clusters as if they had been interpreted.
    """
    return LandscapeSynthesis(
        title=topic,
        summary=(
            f"{len(papers)} papers were retrieved and grouped into "
            f"{len(clusters)} cluster(s). Narrative synthesis was not produced."
        ),
        clusters=list(clusters),
    )


def synthesize_landscape(
    topic: str,
    clusters: Sequence[ClusterLabel],
    members_by_label: dict[int, list[Paper]],
    papers: Sequence[Paper],
    extractions: dict[str, PaperExtraction],
    completer: JSONCompleter | None,
    settings: Settings,
) -> LandscapeSynthesis:
    """Produce the landscape, with cluster names and validated references."""
    if completer is None:
        return fallback_synthesis(topic, clusters, papers)

    rank_of = {paper.paper_id: index + 1 for index, paper in enumerate(papers)}
    sizes = {label: len(members) for label, members in members_by_label.items()}
    cluster_summary = [
        (cluster.local_label, cluster.label, cluster.description, sizes.get(cluster.local_label, 0))
        for cluster in clusters
    ]
    payload = [(paper, extractions.get(paper.paper_id)) for paper in papers]

    result = completer.complete_json(
        system=SYNTHESIS_SYSTEM_PROMPT,
        user=build_synthesis_user_prompt(topic, cluster_summary, payload, rank_of=rank_of),
        schema=LandscapeSynthesis,
        temperature=0.0,
    )
    if result is None:
        logger.warning("Synthesis failed; falling back to the computed structure")
        return fallback_synthesis(topic, clusters, papers)

    valid_ids = {paper.paper_id for paper in papers}
    result = validate_synthesis(result, valid_ids)
    # The computed cluster labels are authoritative; the model only names them.
    return result.model_copy(update={"clusters": list(clusters)})


def group_by_cluster(
    paper_ids: Sequence[str], labels: dict[str, int]
) -> dict[int, list[str]]:
    """Invert ``paper_id -> label`` into ``label -> paper_ids``, order preserved."""
    groups: dict[int, list[str]] = {}
    for paper_id in paper_ids:
        groups.setdefault(int(labels.get(paper_id, UNCLUSTERED_LABEL)), []).append(
            paper_id
        )
    return groups


def run_with_progress(
    fn: Callable[[], LandscapeSynthesis], on_progress: Callable[[int, int], None] | None
) -> LandscapeSynthesis:
    """Tiny helper so callers can report a single synthesis tick uniformly."""
    if on_progress is not None:
        on_progress(0, 1)
    result = fn()
    if on_progress is not None:
        on_progress(1, 1)
    return result


__all__ = [
    "MAX_EDGES",
    "fallback_synthesis",
    "group_by_cluster",
    "label_clusters",
    "placeholder_label",
    "run_with_progress",
    "synthesize_landscape",
    "validate_synthesis",
]
