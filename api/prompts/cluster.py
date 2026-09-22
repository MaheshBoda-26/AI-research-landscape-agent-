"""Cluster naming prompt.

One call per cluster, over the *extracted* problem and contribution of that
cluster's members rather than their raw abstracts. That keeps the prompt small
(so a 30-paper cluster still fits comfortably) and, more importantly, makes the
label a summary of what the papers actually do rather than of how they are
phrased.

The label is descriptive of the shared research direction. It is not a judgement
about quality, and it is not allowed to invent a name that does not follow from
the members supplied.
"""

from __future__ import annotations

from models import Paper, PaperExtraction

#: Per-paper context handed to the labeller.
MAX_FIELD_CHARS = 240


def _condense(extraction: PaperExtraction | None) -> str:
    """One line of extracted structure, or a note that extraction failed."""
    if extraction is None or not any(
        getattr(extraction, field, None) for field in ("problem", "method", "contribution")
    ):
        return "no extraction available"
    parts = []
    for field in ("problem", "contribution", "method"):
        value = getattr(extraction, field, None)
        if value:
            parts.append(f"{field}: {' '.join(str(value).split())[:MAX_FIELD_CHARS]}")
    return " | ".join(parts)


CLUSTER_LABEL_SYSTEM_PROMPT = """You name clusters of research papers on a topic map.

You are given a TOPIC and the members of ONE cluster, each with its extracted
problem, contribution, and method. Return JSON:

{"label": "...", "description": "...", "representative_paper_ids": ["...", "..."]}

Rules for "label":
- 3 to 8 words, in the style of a research area name, not a sentence.
- Name the shared research direction the members have in common.
- Do not include the topic itself verbatim in every label; the point is to
  distinguish this cluster from the other clusters.
- Title case. No trailing period.

Rules for "description":
- One or two sentences on what unifies these papers and how they differ.

Rules for "representative_paper_ids":
- 2 to 4 ids copied exactly from the members listed, picking the most central.
- Never invent an id.

Respond with JSON only."""


def build_cluster_user_prompt(
    topic: str,
    local_label: int,
    members: list[tuple[Paper, PaperExtraction | None]],
) -> str:
    lines = [
        f"TOPIC: {topic}",
        "",
        f"CLUSTER (internal id {local_label}) — {len(members)} paper(s):",
    ]
    for paper, extraction in members:
        title = " ".join(paper.title.split())
        lines.append(f"- [{paper.paper_id}] {title}\n  {_condense(extraction)}")
    return "\n".join(lines)
