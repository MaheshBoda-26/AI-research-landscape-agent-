"""Stage 1 — retrieval.

Turns a plain-English topic into an arXiv query, fetches candidates, and caches
the raw result set on disk.

Three things here are deliberate and load-bearing:

1. **GET only.** ``arxiv.Client`` issues GETs. This matters: the arxiv.py
   maintainer reported that POST requests to ``/api/query`` bypass Fastly's
   cache and therefore hit 429 far sooner. Changing this would break the tool
   under arXiv's post-Feb-2026 rate limiting.
2. **Disk cache with a TTL.** arXiv asks callers to cache rather than repeat
   identical queries, and it is also the only way re-running a demo is fast.
   Every candidate set is written to ``data/cache/arxiv/`` keyed by the exact
   query string.
3. **Version-stripped dedup.** ``arxiv.Result.__eq__`` compares ``entry_id``,
   which includes the ``vN`` suffix. Without stripping, v1 and v3 of the same
   paper are two different results and both appear on the map.

``fetch_candidates`` is synchronous by design because ``arxiv.Client`` wraps a
``requests.Session``. Async callers must wrap it in ``asyncio.to_thread``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import arxiv

from config import Settings
from models import Paper, arXivQuery
from prompts.query import QUERY_SYSTEM, VALID_CATEGORIES, build_query_user

logger = logging.getLogger(__name__)

VERSION_SUFFIX = re.compile(r"^(?P<base>.+?)v(?P<version>\d+)$")


class RetrievalError(RuntimeError):
    """arXiv could not be reached, or offline mode had nothing cached."""


class QueryResolver(Protocol):
    """Minimal surface ``resolve_query`` needs from the LLM client.

    Declared structurally so retrieval does not import the LLM layer; the real
    ``llm.client.LLMClient`` satisfies this without any coupling.
    """

    def complete_json(self, *, system: str, user: str, schema: type) -> Any | None: ...


# --------------------------------------------------------------------------- #
# Query construction
# --------------------------------------------------------------------------- #


def build_query(query: arXivQuery) -> str:
    """Render an ``arXivQuery`` as arXiv API query syntax.

    Phrases are quoted so multi-word terms match as phrases rather than a bag of
    words; keywords are unquoted so their morphology can vary. Category filters
    are ANDed onto the OR'd term group.
    """
    terms: list[str] = []
    for phrase in query.phrases:
        cleaned = " ".join(str(phrase).split())
        if cleaned:
            terms.append(f'all:"{cleaned}"')
    for keyword in query.keywords:
        cleaned = " ".join(str(keyword).split())
        if cleaned:
            terms.append(f"all:{cleaned}")

    if not terms:
        return ""

    joined = " OR ".join(terms)
    categories = [c.strip() for c in query.categories if str(c).strip()]
    if categories:
        return f"({joined}) AND (" + " OR ".join(f"cat:{c}" for c in categories) + ")"
    return joined


def _sanitize_query(raw: arXivQuery | None, topic: str) -> arXivQuery:
    """Drop hallucinated categories and dedupe, keeping the topic as a fallback."""
    if raw is None:
        return arXivQuery(phrases=[topic])

    categories = [c for c in raw.categories if c in VALID_CATEGORIES]
    dropped = [c for c in raw.categories if c not in VALID_CATEGORIES]
    if dropped:
        # Loud, because a bad category filter removes correct papers silently.
        logger.warning("Dropping unrecognized arXiv categories: %s", dropped)

    phrases: list[str] = []
    for phrase in raw.phrases:
        cleaned = " ".join(str(phrase).split())
        if cleaned and cleaned.lower() not in {p.lower() for p in phrases}:
            phrases.append(cleaned)

    keywords: list[str] = []
    for keyword in raw.keywords:
        cleaned = " ".join(str(keyword).split())
        if cleaned and cleaned.lower() not in {k.lower() for k in keywords}:
            keywords.append(cleaned)

    # Cap the term list so the query string stays a reasonable length.
    phrases = phrases[:4]
    keywords = keywords[:5]
    categories = categories[:3]

    if not phrases and not keywords:
        return arXivQuery(phrases=[topic], categories=categories)
    return arXivQuery(phrases=phrases, keywords=keywords, categories=categories)


def resolve_query(topic: str, llm: QueryResolver | None = None) -> arXivQuery:
    """Expand a topic into arXiv search terms, falling back to a quoted topic.

    The fallback matters more than the LLM path: if the model is unavailable the
    run should still search sensibly for the user's literal words.
    """
    fallback = arXivQuery(phrases=[topic])
    if llm is None:
        return fallback
    try:
        result = llm.complete_json(
            system=QUERY_SYSTEM, user=build_query_user(topic), schema=arXivQuery
        )
    except Exception as exc:  # noqa: BLE001 - any LLM failure falls back
        logger.warning("Query expansion failed (%s); using the literal topic", exc)
        return fallback

    if result is None:
        return fallback

    resolved = _sanitize_query(result, topic)
    return resolved if build_query(resolved) else fallback


# --------------------------------------------------------------------------- #
# Result normalization
# --------------------------------------------------------------------------- #


def to_paper(result: arxiv.Result) -> Paper:
    """Normalize one ``arxiv.Result`` into our ``Paper``.

    The short id keeps its version suffix (``2411.18583v1``); the stored
    ``paper_id`` does not. Version-aware callers use ``version``.
    """
    short_id = result.get_short_id()
    match = VERSION_SUFFIX.match(short_id)
    if match:
        base_id = match.group("base")
        version = match.group("version")
    else:
        base_id = short_id
        version = ""

    pdf_url = result.pdf_url or ""
    # Prefer the canonical https://arxiv.org/abs/<id> form: the feed's <id> is
    # plain http, and arXiv asks that users be pointed at the abstract page.
    abs_url = f"https://arxiv.org/abs/{base_id}"

    return Paper(
        paper_id=base_id,
        version=version,
        title=" ".join((result.title or "").split()),
        abstract=" ".join((result.summary or "").split()),
        authors=[a.name for a in (result.authors or [])],
        published=_iso(result.published),
        updated=_iso(result.updated),
        primary_category=result.primary_category or "",
        categories=list(result.categories or []),
        comment=(result.comment or "").strip(),
        journal_ref=(getattr(result, "journal_ref", "") or "").strip(),
        doi=(getattr(result, "doi", "") or "").strip(),
        abs_url=abs_url,
        pdf_url=pdf_url.replace("http://", "https://"),
    )


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return ""


def dedupe_by_paper_id(results: list[arxiv.Result]) -> list[Paper]:
    """Collapse versions of the same paper, keeping the newest.

    Ordering is preserved on first appearance so the API's relevance ordering is
    not disturbed.
    """
    best: dict[str, Paper] = {}
    order: list[str] = []

    for result in results:
        paper = to_paper(result)
        if paper.paper_id not in best:
            best[paper.paper_id] = paper
            order.append(paper.paper_id)
            continue

        current = best[paper.paper_id]
        if _version_number(paper.version) > _version_number(current.version) or (
            paper.version == current.version and paper.updated > current.updated
        ):
            best[paper.paper_id] = paper

    return [best[paper_id] for paper_id in order]


def _version_number(version: str) -> int:
    try:
        return int(version)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def cache_key(query: str, max_results: int) -> str:
    return hashlib.sha256(f"{query}|{max_results}".encode()).hexdigest()


def _cache_path(settings: Settings, key: str) -> Path:
    return Path(settings.retrieval_cache_dir) / f"{key}.json"


def read_cache(settings: Settings, key: str) -> list[Paper] | None:
    """Return cached papers if present and within the TTL, else ``None``."""
    path = _cache_path(settings, key)
    if not path.exists():
        return None
    ttl = settings.retrieval_cache_ttl_hours * 3600
    if ttl > 0 and (time.time() - path.stat().st_mtime) > ttl:
        logger.info("Cache entry %s is older than %dh; refetching", key[:12], settings.retrieval_cache_ttl_hours)
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable cache entry %s: %s", key[:12], exc)
        return None
    try:
        return [Paper(**item) for item in payload.get("papers", [])]
    except Exception as exc:  # noqa: BLE001 - a stale schema must not be fatal
        logger.warning("Ignoring cache entry with an outdated schema (%s)", exc)
        return None


def write_cache(settings: Settings, key: str, query: str, papers: list[Paper]) -> None:
    path = _cache_path(settings, key)
    try:
        Path(settings.retrieval_cache_dir).mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                    "query": query,
                    "max_results": len(papers),
                    "papers": [p.model_dump() for p in papers],
                },
                ensure_ascii=False,
            )
        )
    except OSError as exc:
        # A read-only cache directory must not fail the run.
        logger.warning("Could not write cache entry %s: %s", key[:12], exc)


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


def make_client(settings: Settings) -> arxiv.Client:
    """Build the arXiv client with the configured rate limiting.

    ``delay_seconds`` is enforced by the library, so nothing here sleeps
    manually — a second delay layer would make runs unusably slow.
    """
    return arxiv.Client(
        page_size=settings.arxiv_page_size,
        delay_seconds=settings.arxiv_delay_seconds,
        num_retries=settings.arxiv_num_retries,
    )


def fetch_candidates(
    topic: str,
    settings: Settings,
    *,
    llm: QueryResolver | None = None,
    client: arxiv.Client | None = None,
    use_cache: bool = True,
) -> list[Paper]:
    """Fetch candidate papers for a topic. Synchronous; see the module docstring.

    Raises ``RetrievalError`` when arXiv is unreachable and nothing is cached,
    with a message that names throttling when that is what happened.
    """
    query = build_query(resolve_query(topic, llm))
    if not query:
        raise RetrievalError(f"Could not build an arXiv query for topic: {topic!r}")

    key = cache_key(query, settings.retrieval_max_results)
    if use_cache:
        cached = read_cache(settings, key)
        if cached is not None:
            logger.info("Cache hit for %r: %d papers", topic, len(cached))
            return cached

    if client is None:
        # Only guard the real network path. An injected client (tests) makes no
        # requests, so it must not be blocked by the offline flag.
        if settings.arxiv_offline:
            raise RetrievalError(
                "ARXIV_OFFLINE is set and no cached result exists for this topic. "
                "Unset it (or warm the cache) to fetch from arXiv."
            )
        client = make_client(settings)
    search = arxiv.Search(
        query=query,
        max_results=settings.retrieval_max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    try:
        results = list(client.results(search))
    except (arxiv.UnexpectedEmptyPageError, arxiv.HTTPError) as exc:
        raise RetrievalError(_explain(exc)) from exc

    if not results:
        # Graceful degradation: the expanded query may simply be too narrow.
        literal = f'all:"{topic}"'
        if literal != query:
            logger.info("Expanded query returned nothing; retrying the literal topic")
            try:
                results = list(
                    client.results(
                        arxiv.Search(
                            query=literal,
                            max_results=settings.retrieval_max_results,
                            sort_by=arxiv.SortCriterion.Relevance,
                        )
                    )
                )
            except (arxiv.UnexpectedEmptyPageError, arxiv.HTTPError) as exc:
                raise RetrievalError(_explain(exc)) from exc

    papers = dedupe_by_paper_id(results)
    if use_cache and papers:
        write_cache(settings, key, query, papers)
    logger.info("Retrieved %d papers for %r", len(papers), topic)
    return papers


def _explain(exc: Exception) -> str:
    """Turn a library exception into something a user can act on."""
    status = getattr(exc, "status", None)
    if status == 429:
        return (
            "arXiv is throttling this client (HTTP 429). arXiv tightened its rate "
            "limits in 2026 and routinely rejects requests that obey the documented "
            "1-per-3-seconds rule. Wait a few minutes and retry; cached topics still work."
        )
    if status in (500, 502, 503, 504):
        return f"arXiv returned HTTP {status}. This is server-side; retry shortly."
    return f"arXiv request failed: {exc}"
