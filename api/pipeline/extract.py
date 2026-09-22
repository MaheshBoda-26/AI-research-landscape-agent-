"""Stage 3 — structured extraction.

One LLM call per paper turns an abstract into problem, method, results,
contribution, limitations, plus datasets, metrics, and a novelty assessment.

The interesting part is the evidence check. Every populated prose field must be
backed by a quote from the abstract, and that quote is verified as a literal
substring before the field is stored. A field that cannot be grounded is nulled
rather than displayed, because a map that describes papers inaccurately is worse
than a map with holes in it.

This mirrors the citation-verification discipline used in the RAG-Pipeline
project's ``generation/citations.py``: generated text is only trusted once it has
been checked against its source, and the check is lexical rather than a second
model's opinion.

Concurrency is a thread pool sized to ``LLM_CONCURRENCY``, matching the client's
own semaphore. Caching is keyed on ``(paper_id, prompt_version, model)``, so
iterating on the prompt never forces a re-embed or a re-layout.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from config import Settings
from llm.protocol import JSONCompleter
from models import Paper, PaperExtraction
from prompts.extract import (
    EXTRACT_SYSTEM_PROMPT,
    build_correction_prompt,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

#: A quote this short is a substring by accident, not evidence.
MIN_EVIDENCE_CHARS = 16

#: Characters that vary by typography rather than by meaning. Abstracts mix
#: LaTeX, Unicode dashes, and smart quotes, so a verbatim quote frequently fails
#: a naive substring test for reasons that have nothing to do with the model.
_PUNCTUATION_MAP = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
    "\u2009": " ",
    "\u202f": " ",
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Normalize text for substring comparison, preserving wording exactly."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    translated = folded.translate(str.maketrans({k: v for k, v in _PUNCTUATION_MAP.items()}))
    collapsed = _WHITESPACE_RE.sub(" ", translated)
    return collapsed.strip().casefold()


def is_verbatim(quote: str | None, abstract: str) -> bool:
    """True when ``quote`` appears in ``abstract``, ignoring typography only."""
    if not quote or len(quote.strip()) < MIN_EVIDENCE_CHARS:
        return False
    return normalize_for_match(quote) in normalize_for_match(abstract)


def ungrounded_fields(extraction: PaperExtraction, abstract: str) -> list[str]:
    """Names of populated prose fields whose evidence quote is not verbatim.

    A non-null field with no evidence entry at all is also ungrounded: the
    contract requires a quote, and silence is not a quote.
    """
    bad: list[str] = []
    for field in PaperExtraction.EXTRACTED_FIELDS:
        value = getattr(extraction, field, None)
        if value is None or not str(value).strip():
            continue
        if not is_verbatim(extraction.evidence.get(field), abstract):
            bad.append(field)
    return bad


def enforce_evidence(extraction: PaperExtraction, abstract: str) -> PaperExtraction:
    """Null any field that could not be grounded, and drop stale evidence keys.

    Also demotes ``novelty`` to ``"unclear"`` when nothing could be grounded: a
    novelty judgement with no supporting text is exactly the kind of confident
    unsupported claim this pipeline exists to avoid.
    """
    bad = set(ungrounded_fields(extraction, abstract))
    if not bad:
        return extraction

    updates: dict[str, object] = dict(extraction.model_dump())
    for field in bad:
        updates[field] = None
    updates["evidence"] = {
        key: value for key, value in extraction.evidence.items() if key not in bad
    }
    grounded = sum(
        1
        for field in PaperExtraction.EXTRACTED_FIELDS
        if getattr(extraction, field) and field not in bad
    )
    if grounded == 0:
        updates["novelty"] = "unclear"
        updates["confidence"] = None
    elif extraction.confidence is not None:
        # Penalise proportionally to how much of the extraction was invented.
        total = sum(1 for f in PaperExtraction.EXTRACTED_FIELDS if getattr(extraction, f))
        if total:
            updates["confidence"] = round(extraction.confidence * (grounded / total), 4)

    logger.info("Ungrounded fields nulled for a paper: %s", sorted(bad))
    return PaperExtraction(**updates)  # type: ignore[arg-type]


def unextracted(paper_id: str = "") -> PaperExtraction:
    """The stored result of a failed extraction.

    An explicit nulled row rather than a missing one, so a re-run can tell
    "attempted and failed" apart from "never attempted" without another call.
    """
    return PaperExtraction(novelty="unclear", confidence=None)


def extract_one(
    paper: Paper,
    completer: JSONCompleter | None,
    settings: Settings,
    *,
    corrective_retries: int = 1,
) -> PaperExtraction:
    """Extract structure from one abstract, grounding every field."""
    if completer is None:
        return unextracted(paper.paper_id)

    user = build_user_prompt(paper)
    result = completer.complete_json(
        system=EXTRACT_SYSTEM_PROMPT,
        user=user,
        schema=PaperExtraction,
        temperature=0.0,
    )
    if result is None:
        logger.warning("Extraction failed for %s; storing a null record", paper.paper_id)
        return unextracted(paper.paper_id)

    bad = ungrounded_fields(result, paper.abstract)
    attempts = 0
    while bad and attempts < corrective_retries:
        attempts += 1
        logger.info("Re-prompting for %s: %d ungrounded field(s)", paper.paper_id, len(bad))
        result = completer.complete_json(
            system=EXTRACT_SYSTEM_PROMPT,
            user=f"{user}\n\n{build_correction_prompt(bad)}",
            schema=PaperExtraction,
            temperature=0.0,
        )
        if result is None:
            return unextracted(paper.paper_id)
        bad = ungrounded_fields(result, paper.abstract)

    return enforce_evidence(result, paper.abstract)


def extract_all(
    papers: Sequence[Paper],
    completer: JSONCompleter | None,
    settings: Settings,
    conn: sqlite3.Connection | None = None,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, PaperExtraction]:
    """Extract structure for ``papers``, using the cache and a thread pool.

    Returns ``paper_id -> extraction``. Papers already extracted under this
    ``(prompt_version, model)`` are not sent to the model again.
    """
    if not papers:
        return {}

    prompt_version = settings.prompt_version
    cached: dict[str, PaperExtraction] = {}
    if conn is not None:
        from store import fetch_extractions

        cached = fetch_extractions(conn, [p.paper_id for p in papers], prompt_version)

    todo = [p for p in papers if p.paper_id not in cached]
    logger.info(
        "Extraction: %d cached, %d to run (prompt_version=%s)",
        len(cached),
        len(todo),
        prompt_version,
    )

    if not todo or completer is None:
        if completer is None and todo:
            logger.warning("No LLM client; %d papers will be stored unextracted", len(todo))
            for paper in todo:
                cached[paper.paper_id] = unextracted(paper.paper_id)
        if on_progress is not None:
            on_progress(len(papers), len(papers))
        return cached

    completed = len(cached)
    total = len(papers)
    if on_progress is not None:
        on_progress(completed, total)

    # Ordered results so the DB write order is deterministic.
    ordered_ids = [p.paper_id for p in todo]
    results: dict[str, PaperExtraction] = {}

    def work(paper: Paper) -> tuple[str, PaperExtraction]:
        return paper.paper_id, extract_one(paper, completer, settings)

    workers = max(1, min(settings.llm_concurrency, len(todo)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="extract") as pool:
        for paper_id, extraction in pool.map(work, todo):
            results[paper_id] = extraction
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)

    if conn is not None:
        from store import upsert_extraction

        model = getattr(completer, "model_name", "") or "unknown"
        for paper_id in ordered_ids:
            upsert_extraction(
                conn, paper_id, prompt_version, results[paper_id], model=str(model)
            )
        conn.commit()

    merged = dict(cached)
    merged.update(results)
    return merged


__all__ = [
    "MIN_EVIDENCE_CHARS",
    "enforce_evidence",
    "extract_all",
    "extract_one",
    "is_verbatim",
    "normalize_for_match",
    "unextracted",
    "ungrounded_fields",
]
