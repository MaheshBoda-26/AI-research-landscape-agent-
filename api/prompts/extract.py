"""Structured extraction prompt.

The extraction contract is where this project either produces a trustworthy map
or a plausible-sounding fiction. Two rules do the heavy lifting:

1. **Every populated field must be backed by a verbatim quote** from the
   abstract, returned in the ``evidence`` map. The quote is verified as a real
   substring before the field is stored, so a fabricated claim is rejected
   rather than displayed.
2. **Use ``null`` when the abstract does not say.** Most abstracts do not state
   limitations, and inventing one is the single most likely hallucination in
   this task. An absent value is a correct answer.

Bump the ``PROMPT_VERSION`` environment variable whenever this prompt or the
``PaperExtraction`` schema changes, so cached extractions are recomputed.
"""

from __future__ import annotations

from models import Paper

MAX_ABSTRACT_CHARS = 6000

EXTRACT_SYSTEM_PROMPT = """You read a machine-learning paper abstract and extract its structure.

Return JSON with exactly these keys:

- "problem": what gap or question the paper addresses (1 sentence)
- "method": what they actually built or did (1-2 sentences)
- "results": concrete outcomes, including numbers if the abstract gives them (1-2 sentences)
- "contribution": the paper's main claimed contribution (1 sentence)
- "limitations": limitations the abstract itself states, or null
- "datasets": array of dataset names mentioned, [] if none
- "metrics": array of metric names mentioned, [] if none
- "novelty": one of "incremental", "substantial", "unclear"
- "evidence": object mapping each of the five prose field names above to a
  VERBATIM substring copied exactly from the abstract that supports it
- "confidence": your confidence from 0.0 to 1.0 that the extraction is faithful

Absolute rules:
- Use ONLY information present in the abstract. Never use outside knowledge of
  the paper, its authors, or its field.
- Every string in "evidence" must appear character-for-character in the abstract.
  Do not paraphrase, correct, or shorten it.
- If the abstract does not state something, use null for that field (or [] for
  the arrays). Do not guess, and do not write "not stated" — use null.
- "limitations" is null for most papers. That is expected and correct.
- Do not include a field in "evidence" if that field is null.
- Respond with JSON only, no prose and no code fences."""


def build_user_prompt(paper: Paper) -> str:
    abstract = paper.abstract[:MAX_ABSTRACT_CHARS]
    return f"TITLE: {paper.title}\n\nABSTRACT:\n{abstract}"


def build_correction_prompt(bad_fields: list[str]) -> str:
    """Second-pass instruction when evidence quotes were not verbatim."""
    listed = ", ".join(sorted(bad_fields))
    return (
        "Your previous response failed verification: the evidence for these fields "
        f"was not a verbatim substring of the abstract: {listed}. "
        "Re-read the abstract and return the full JSON object again. For each field, "
        "copy the supporting text exactly as written, character for character. "
        "If you cannot find a verbatim quote for a field, set that field to null "
        "and omit it from evidence. Respond with JSON only."
    )
