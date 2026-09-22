"""LLM-as-judge reranking prompt.

Keyword overlap is not relevance: a paper can match every term in a topic and
still be a passing mention. The cross-encoder handles most of this, but it is
trained on web passages and is unreliable for research prose, so a judge that
reads the topic and each abstract is used to correct the top of the ranking.

The judge returns scores on the same 0-10 scale the cross-encoder produces, so
blending them needs no rescaling and every downstream consumer sees one scale.
It answers with a JSON *object* (``{"scores": [...]}``) rather than a bare array
so that the response validates against a Pydantic model like every other call.
"""

from __future__ import annotations

from models import Paper

#: Passages are truncated before they reach the judge. Abstracts run 1200-1900
#: characters; 1500 covers the whole of nearly all of them while keeping a
#: ten-paper batch inside a comfortable context budget.
MAX_PASSAGE_CHARS = 1500

JUDGE_SYSTEM_PROMPT = """You are a relevance judge for an academic search engine.

You are given a TOPIC and a numbered list of paper abstracts. For each paper,
score how relevant it is as a starting point for learning that topic, from 0 to 10:

- 10 = the paper is centrally about the topic; reading it is essential
- 7-9 = the paper substantially addresses the topic
- 4-6 = the paper is topically adjacent but not about the topic itself
- 1-3 = the paper merely mentions the topic in passing
- 0 = the paper is unrelated

Judge how central the topic is to the paper, not how impressive the paper is.
A famous paper that only mentions the topic scores lower than an obscure paper
that is entirely about it. Surveys, benchmarks, and position papers are all
legitimate high scorers when they are about the topic.

Respond with JSON only, in exactly this shape:
{"scores": [{"id": 1, "score": 8, "reason": "one short clause"}, {"id": 2, "score": 2, "reason": "..."}]}

Include every id you were given, exactly once, and no others."""


def build_judge_user_prompt(topic: str, papers: list[Paper]) -> str:
    """Numbered passage block. The numbering is the alignment contract."""
    blocks = []
    for index, paper in enumerate(papers, start=1):
        text = paper.rerank_text[:MAX_PASSAGE_CHARS]
        blocks.append(f"[{index}] {text}")
    joined = "\n\n".join(blocks)
    return f"TOPIC: {topic}\n\nABSTRACTS:\n\n{joined}"
