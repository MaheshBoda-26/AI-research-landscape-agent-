#!/usr/bin/env python
"""Headless pipeline driver.

Every stage is runnable from here before any UI exists, which is the only
practical way to iterate on prompt quality. Debugging extraction through a
browser is how this kind of project stalls.

Usage::

    python scripts/run_pipeline.py --topic "retrieval-augmented generation" --stage retrieve
    python scripts/run_pipeline.py --topic "..." --stage rerank
    python scripts/run_pipeline.py --topic "..." --all

Run ``--help`` for the full surface.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the api package importable when this script is run from the repo root.
API_ROOT = Path(__file__).resolve().parent.parent / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from config import ConfigError, Settings, load_settings  # noqa: E402
from models import Paper, RankedPaper  # noqa: E402
from pipeline import rerank, retrieve  # noqa: E402

STAGES = ("retrieve", "rerank", "extract", "layout", "synthesize")


def _truncate(text: str, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def print_papers(papers: list[Paper], *, limit: int, show_score: bool = False) -> None:
    if not papers:
        print("  (no papers)")
        return
    print(f"  {'#':>3}  {'score':>6}  {'id':<14}  title" if show_score else f"  {'#':>3}  {'id':<14}  title")
    for index, paper in enumerate(papers[:limit], start=1):
        prefix = f"  {index:>3}"
        if show_score:
            prefix += f"  {getattr(paper, 'relevance_score', 0.0):>6.2f}"
        print(f"{prefix}  {paper.paper_id:<14}  {_truncate(paper.title, 68)}")
    if len(papers) > limit:
        print(f"  ... and {len(papers) - limit} more")


def stage_retrieve(args, settings: Settings) -> list[Paper]:
    print(f"\n[retrieve] searching arXiv for {args.topic!r}")
    papers = retrieve.fetch_candidates(
        args.topic, settings, force_refresh=args.refresh
    )
    print(f"[retrieve] {len(papers)} unique papers after version dedup\n")
    print_papers(papers, limit=args.limit)
    return papers


def stage_rerank(args, settings: Settings, papers: list[Paper]) -> list[RankedPaper]:
    completer = make_completer(settings) if args.llm else None
    print(f"\n[rerank] scoring {len(papers)} candidates")
    if completer is None:
        print("[rerank] no LLM: cross-encoder only (pass --llm to blend in the judge)")
    ranked = rerank.rank_papers(args.topic, papers, settings, completer)

    counts: dict[str, int] = {}
    for item in ranked:
        counts[item.rerank_source] = counts.get(item.rerank_source, 0) + 1
    print(f"[rerank] sources: {counts}\n")

    final = rerank.select_top(ranked, settings.rerank_final_count)
    # The calibrated score (abs) saturates near 10 for a retrieved candidate set,
    # so display the raw logit and the relative percentile alongside it. Without
    # these two columns every row reads "10.00" and the ranking looks arbitrary.
    print(f"  {'#':>3}  {'abs':>6}  {'logit':>6}  {'rel':>5}  {'src':<13}  {'id':<14}  title")
    for item in final[: args.limit]:
        logit = "-" if item.cross_encoder_logit is None else f"{item.cross_encoder_logit:.2f}"
        rel = "-" if item.relative_score is None else f"{item.relative_score:.1f}"
        print(
            f"  {item.rank:>3}  {item.relevance_score:>6.2f}  {logit:>6}  {rel:>5}  "
            f"{item.rerank_source:<13}  {item.paper.paper_id:<14}  "
            f"{_truncate(item.paper.title, 40)}"
        )
    return final


def make_completer(settings: Settings):
    """Build the LLM client, or explain why we are running without one."""
    try:
        from llm.client import LLMClient
    except ImportError:
        print("[llm] client not available; running without it", file=sys.stderr)
        return None
    if not settings.llm_api_key:
        key_var = "NVIDIA_API_KEY" if settings.llm_provider == "nim" else "OPENROUTER_API_KEY"
        print(f"[llm] {key_var} is unset; running without an LLM", file=sys.stderr)
        return None
    return LLMClient(settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the research landscape pipeline from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topic", required=True, help="Plain-English ML topic.")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="retrieve",
        help="Which stage to run and print (default: retrieve).",
    )
    parser.add_argument("--all", action="store_true", help="Run every stage in order.")
    parser.add_argument("--limit", type=int, default=15, help="Rows to print (default: 15).")
    parser.add_argument("--refresh", action="store_true", help="Bypass the fresh arXiv cache.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--offline", action="store_true", help="Never touch the network.")
    parser.add_argument("--llm", action="store_true", help="Use the configured LLM (query expansion etc.).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = load_settings(require_llm=args.llm)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    if args.offline:
        from dataclasses import replace

        settings = replace(settings, arxiv_offline=True)

    papers = stage_retrieve(args, settings)
    ranked: list[RankedPaper] = []

    if args.all or args.stage in {"rerank", "extract", "layout", "synthesize"}:
        ranked = stage_rerank(args, settings, papers)

    if args.json:
        payload = [r.model_dump() for r in ranked] if ranked else [p.model_dump() for p in papers]
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
