"""Landscape synthesis prompt.

The one rule that matters: the model may only reference paper ids that were
supplied. Everything else here is about quality — a tension must be a real
disagreement, not a restatement that two papers differ; an open problem must
follow from what the papers left undone, not from the model's general knowledge
of the field.

References are validated after parsing, so a non-compliant id is dropped rather
than trusted, but the prompt states the rule anyway because a compliant model
produces a much better map than a filtered non-compliant one.
"""

from __future__ import annotations

from models import Paper, PaperExtraction

MAX_FIELD_CHARS = 260
MAX_PAPERS_IN_PROMPT = 60


def _line(paper: Paper, extraction: PaperExtraction | None, rank: int) -> str:
    title = " ".join(paper.title.split())[:160]
    if extraction is None or not any(
        getattr(extraction, field, None) for field in ("problem", "contribution")
    ):
        return f"- [{paper.paper_id}] (rank {rank}) {title}\n  no extraction available"
    bits = []
    for field in ("problem", "method", "contribution", "limitations"):
        value = getattr(extraction, field, None)
        if value:
            bits.append(f"{field}: {' '.join(str(value).split())[:MAX_FIELD_CHARS]}")
    return f"- [{paper.paper_id}] (rank {rank}) {title}\n  " + " | ".join(bits)


SYNTHESIS_SYSTEM_PROMPT = """You write the narrative layer of a research landscape map.

You are given a topic, a set of computed clusters with their names, and the
papers in those clusters with their extracted structure. Return JSON:

{"title": "...", "summary": "...", "edges": [...], "tensions": [...],
 "open_problems": [...], "reading_path": [...]}

"title": a specific title for this landscape, 4-10 words. Not the topic verbatim.

"summary": 3-5 sentences describing the shape of the field: what the main
clusters are, what the field has largely settled, and where it is still moving.

"edges": relationships between papers. Each entry:
  {"src_paper_id": "...", "dst_paper_id": "...", "kind": "...", "weight": 0.0-1.0,
   "rationale": "one short clause"}
  "kind" is exactly one of: "extends", "contradicts", "applies", "shares_method".
  Use "contradicts" ONLY when the two papers' stated findings or claims actually
  conflict. Being about different approaches is not a contradiction.
  Use "extends" when dst builds directly on src. Prefer a few strong edges
  (10-25 total) over many weak ones.

"tensions": genuine disagreements in the field. Each entry:
  {"statement": "...", "paper_a_id": "...", "paper_b_id": "..."}
  A tension must name two supplied papers whose stated results actually conflict.
  If the papers do not disagree, return an empty list. Do NOT manufacture a
  tension to fill the section.

"open_problems": unanswered questions that follow from what these papers left
undone. Each entry:
  {"statement": "...", "why_open": "...", "supporting_paper_ids": ["..."]}

"reading_path": the order a newcomer should read these papers in. Each entry:
  {"paper_id": "...", "position": 1, "why": "one short clause"}
  Start with the most accessible foundational or survey work, then move to the
  specialised and most recent work. Include 5-10 papers.

ABSOLUTE RULES:
- Every paper id you write must be one of the ids supplied below. Never invent,
  guess, or abbreviate an id.
- Base every claim on the supplied extractions. Do not add knowledge about the
  field from outside this material.
- Respond with JSON only, no prose and no code fences."""


def build_synthesis_user_prompt(
    topic: str,
    clusters: list[tuple[int, str, str, int]],
    papers: list[tuple[Paper, PaperExtraction | None]],
    *,
    rank_of: dict[str, int],
) -> str:
    """Assemble the synthesis context.

    ``clusters`` is ``(local_label, label, description, size)`` and ``papers`` is
    ordered best-first so the strongest material appears early.
    """
    lines = [f"TOPIC: {topic}", "", "CLUSTERS:"]
    for local_label, label, description, size in clusters:
        lines.append(f"- {label} ({size} papers, internal id {local_label}): {description}")

    lines.append("")
    lines.append(f"PAPERS ({min(len(papers), MAX_PAPERS_IN_PROMPT)} supplied):")
    for paper, extraction in papers[:MAX_PAPERS_IN_PROMPT]:
        lines.append(_line(paper, extraction, rank_of.get(paper.paper_id, 0)))

    ids = [paper.paper_id for paper, _ in papers[:MAX_PAPERS_IN_PROMPT]]
    lines.append("")
    lines.append("VALID PAPER IDS (the only ids you may reference):")
    lines.append(", ".join(ids))
    return "\n".join(lines)
