"""Stage 1 — retrieval.

Fetches candidate papers from arXiv for a plain-English topic.

Two external constraints shape this module, and neither is negotiable:

**Rate limits.** The arXiv Terms of Use ask for no more than one request every
three seconds on a single connection. ``arxiv.Client`` enforces that spacing, but
only for a *reused* client — a fresh client has no memory of the previous
request. So exactly one client is built per run and threaded through every
request, including the unquoted-retry fallback.

**Availability.** Since early 2026 arXiv has been returning 429 and 503 to
callers who are under the documented limit, and POST requests bypass the Fastly
cache that makes GET requests cheap to serve. Therefore: GET only (``arxiv.py``
uses GET), every response cached to disk, and a warm-but-stale cache is preferred
over a failed run.

Synchronous by construction because the upstream library is. Callers in async
code must wrap this in ``asyncio.to_thread``.

Test seams
----------
``client`` lets a test inject an ``arxiv.Client`` whose HTTP session is faked,
which exercises the *real* URL formatting, pagination, retry, and Atom-parsing
code without touching the network. ``use_cache=False`` disables both cache
reads and writes, including the stale-fallback path, so failure modes can be
asserted deterministically.
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
import requests

from config import Settings, load_settings
from models import Paper, arXivQuery
from prompts.query import (
    ML_CATEGORIES,
    QUERY_SYSTEM_PROMPT,
    build_query,
    build_user_prompt,
    heuristic_query,
    plain_query,
)

logger = logging.getLogger(__name__)

#: Trailing version marker on an arXiv short id, e.g. ``2411.18583v2``.
_VERSION_RE = re.compile(r"v(\d+)$", re.IGNORECASE)

#: arXiv's sentinel for a missing date.
_MIN_PLAUSIBLE_YEAR = 1900

_SECONDS_PER_HOUR = 3600


class RetrievalError(RuntimeError):
    """arXiv could not be reached, or the request could not be made."""


class RetrievalThrottled(RetrievalError):
    """arXiv refused the request with 429 or 503.

    Distinct from ``RetrievalError`` because it is *retryable*: the SSE error
    frame reports it as such so the UI can say "arXiv is throttling us" instead
    of marking the landscape permanently failed.
    """


class RetrievalOffline(RetrievalError):
    """Offline mode was requested and nothing was cached for this query."""


class QueryResolver(Protocol):
    """The minimal surface ``resolve_query`` needs.

    Declared structurally so ``retrieve`` does not import the LLM package, and
    so tests can pass a two-line fake.
    """

    def complete_json(
        self, *, system: str, user: str, schema: type, **kwargs: Any
    ) -> Any | None: ...


# --------------------------------------------------------------------------- #
# Identifier handling
# --------------------------------------------------------------------------- #


def normalize_paper_id(short_id: str) -> str:
    """Strip the version suffix from an arXiv short id.

    ``arxiv.Result.__eq__`` compares ``entry_id``, which *includes* the version,
    so ``2411.18583v1`` and ``2411.18583v3`` are different results upstream. This
    project keys papers on the version-free id so a revised paper occupies one
    row and one node in the map instead of two.
    """
    return _VERSION_RE.sub("", (short_id or "").strip())


def version_suffix(short_id: str) -> str:
    """Return the numeric version as a string, or ``""`` when absent."""
    match = _VERSION_RE.search((short_id or "").strip())
    return match.group(1) if match else ""


def _version_number(version: str) -> int:
    try:
        return int(version)
    except (TypeError, ValueError):
        return 0


def _iso(value: Any) -> str:
    """Serialize an arXiv date, treating the sentinel minimum as absent."""
    if isinstance(value, datetime):
        return "" if value.year < _MIN_PLAUSIBLE_YEAR else value.isoformat()
    return str(value or "")


# --------------------------------------------------------------------------- #
# Result conversion
# --------------------------------------------------------------------------- #


def result_to_paper(result: arxiv.Result) -> Paper:
    """Convert an ``arxiv.Result`` into the project's ``Paper`` model."""
    short_id = result.get_short_id()
    paper_id = normalize_paper_id(short_id)
    return Paper(
        paper_id=paper_id,
        version=version_suffix(short_id),
        title=(result.title or "").strip(),
        abstract=(result.summary or "").strip(),
        authors=[author.name for author in result.authors or []],
        published=_iso(result.published),
        updated=_iso(result.updated),
        primary_category=result.primary_category or "",
        categories=list(result.categories or []),
        comment=(result.comment or "").strip(),
        journal_ref=(result.journal_ref or "").strip(),
        doi=(result.doi or "").strip(),
        # arXiv asks that users be directed to the abstract page, which is also
        # the only page this project may point at: e-print PDFs must not be
        # re-served from our own origin. pdf_url is kept for linking out.
        abs_url=f"https://arxiv.org/abs/{paper_id}",
        # Not version-stripped: this is the upstream file URL used for linking.
        pdf_url=result.pdf_url or "",
    )


#: Backwards-compatible alias for the name this function had previously.
to_paper = result_to_paper


def dedupe_by_paper_id(papers: list[Paper]) -> list[Paper]:
    """Collapse multiple versions of one paper, keeping the newest version.

    First-appearance order is preserved so arXiv's relevance ordering survives;
    only the *contents* of a duplicate are replaced.
    """
    best: dict[str, Paper] = {}
    order: list[str] = []
    for paper in papers:
        if not paper.paper_id:
            continue
        current = best.get(paper.paper_id)
        if current is None:
            best[paper.paper_id] = paper
            order.append(paper.paper_id)
        elif _version_number(paper.version) > _version_number(current.version):
            best[paper.paper_id] = paper
    return [best[pid] for pid in order]


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #


def cache_key(query: str, max_results: int) -> str:
    """Stable key for a (query, limit) pair."""
    payload = f"{query}|{max_results}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def _cache_path(settings: Settings, key: str) -> Path:
    return settings.retrieval_cache_dir / f"{key}.json"


def write_cache(
    settings: Settings,
    key: str,
    query: str,
    papers: list[Paper],
    *,
    max_results: int | None = None,
) -> None:
    """Persist a response so re-runs and demos never re-hit the API.

    arXiv's own guidance is to cache results locally rather than repeating the
    same query, and the maintainer of ``arxiv.py`` makes the same point: repeated
    identical GETs are what the Fastly cache is for, and a local cache is cheaper
    still.

    Never raises: a cache write failure must not fail a pipeline run.
    """
    path = _cache_path(settings, key)
    payload = {
        "query": query,
        "max_results": len(papers) if max_results is None else max_results,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "papers": [paper.model_dump() for paper in papers],
    }
    try:
        settings.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Could not write arXiv cache %s: %s", path, exc)


def read_cache(settings: Settings, key: str, *, allow_stale: bool = False) -> list[Paper] | None:
    """Load a cached response, or ``None`` when missing or too old.

    Freshness is decided by the file's mtime rather than a field inside the
    payload: mtime cannot drift out of sync with reality, and it is the value the
    filesystem already maintains. ``created_at`` is still written for
    diagnostics. A TTL of zero or less disables expiry entirely.

    ``allow_stale=True`` is the graceful-degradation path: when arXiv is
    throttling, yesterday's results are far more useful than an error.
    """
    path = _cache_path(settings, key)
    if not path.exists():
        return None

    ttl_hours = settings.retrieval_cache_ttl_hours
    if not allow_stale and ttl_hours > 0:
        try:
            age_seconds = time.time() - path.stat().st_mtime
        except OSError:
            return None
        if age_seconds > ttl_hours * _SECONDS_PER_HOUR:
            return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable arXiv cache %s: %s", path, exc)
        return None

    papers: list[Paper] = []
    for raw in payload.get("papers", []):
        try:
            papers.append(Paper(**raw))
        except (TypeError, ValueError):
            # One malformed entry must not discard the whole page.
            continue
    return papers or None


# --------------------------------------------------------------------------- #
# Query resolution
# --------------------------------------------------------------------------- #


def resolve_query(
    topic: str, settings: Settings, completer: QueryResolver | None = None
) -> arXivQuery:
    """Expand a topic into arXiv search terms, falling back to the raw topic.

    Failure here is never fatal: without a completer, or if the model returns
    nothing usable, the topic itself is searched as a quoted phrase.
    """
    fallback = heuristic_query(topic)
    if completer is None:
        return fallback

    try:
        result = completer.complete_json(
            system=QUERY_SYSTEM_PROMPT, user=build_user_prompt(topic), schema=arXivQuery
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never break retrieval
        logger.warning("Query expansion failed; searching the raw topic instead: %s", exc)
        return fallback

    if result is None or not (getattr(result, "phrases", None) or getattr(result, "keywords", None)):
        return fallback

    # Keep only categories that exist in the ML literature. A hallucinated
    # category silently reduces recall to zero, which reads as a broken app
    # rather than a bad model guess.
    allowed = set(ML_CATEGORIES)
    return arXivQuery(
        phrases=list(result.phrases)[:4],
        keywords=list(result.keywords)[:8],
        categories=[c for c in result.categories if c in allowed][:3],
    )


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def make_client(settings: Settings) -> arxiv.Client:
    """Build the arXiv client with the configured rate limiting.

    Reuse one per run so the delay is honoured between successive requests.
    """
    return arxiv.Client(
        page_size=settings.arxiv_page_size,
        delay_seconds=settings.arxiv_delay_seconds,
        num_retries=settings.arxiv_num_retries,
    )


def _fetch(
    client: arxiv.Client,
    query: str,
    max_results: int,
    *,
    simulate_throttle: bool = False,
) -> list[arxiv.Result]:
    """Run one search, translating upstream failures into our error types."""
    if simulate_throttle:
        # Deterministic stand-in for the real thing. It fires only on the real
        # network path, so tests that inject a client are unaffected.
        raise RetrievalThrottled(
            "arXiv is throttling this client (HTTP 429). Set ARXIV_FORCE_429=0 to disable "
            "the simulation."
        )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    try:
        return list(client.results(search))
    except arxiv.HTTPError as exc:
        if exc.status == 429:
            raise RetrievalThrottled(
                "arXiv is throttling this client (HTTP 429). arXiv tightened its rate "
                "limits in 2026, so this is expected under load — cached results are "
                "used when available, and the same topic usually succeeds on a retry."
            ) from exc
        if exc.status == 503:
            raise RetrievalThrottled(
                "arXiv is temporarily unavailable (HTTP 503). This is an upstream outage; "
                "cached results are used when available."
            ) from exc
        raise RetrievalError(f"arXiv returned HTTP {exc.status} for query {query!r}.") from exc
    except arxiv.UnexpectedEmptyPageError as exc:
        raise RetrievalError(
            "arXiv returned an unexpectedly empty page; this is a known upstream "
            "flakiness that usually resolves on retry."
        ) from exc
    except requests.RequestException as exc:
        raise RetrievalError(f"Could not reach arXiv: {exc}") from exc


def fetch_candidates(
    topic: str,
    settings: Settings | None = None,
    *,
    client: arxiv.Client | None = None,
    use_cache: bool = True,
    completer: QueryResolver | None = None,
    force_refresh: bool = False,
) -> list[Paper]:
    """Return deduped candidate papers for ``topic``.

    Preference order: fresh cache -> live fetch -> stale cache -> ``[]``.

    An empty list means the search was valid but matched nothing, which is a
    legitimate outcome the caller should surface as "no papers for this topic".
    A blank topic, by contrast, is an input error and raises.
    """
    settings = settings or load_settings(require_llm=False)

    cleaned_topic = " ".join((topic or "").split())
    if not cleaned_topic:
        raise RetrievalError("A topic is required, but the search string was empty.")

    query = build_query(resolve_query(cleaned_topic, settings, completer))
    fallback_query = plain_query(cleaned_topic)
    key = cache_key(f"{query}||{fallback_query}", settings.retrieval_max_results)

    if use_cache and not force_refresh:
        fresh = read_cache(settings, key)
        if fresh:
            logger.info("Retrieved %d papers from cache for %r", len(fresh), cleaned_topic)
            return fresh

    # An injected client makes no real requests, so the offline flag must not
    # block it: otherwise the network-free tests could not run offline at all.
    using_real_network = client is None

    if using_real_network and settings.arxiv_offline:
        stale = read_cache(settings, key, allow_stale=True) if use_cache else None
        if stale:
            return stale
        raise RetrievalOffline(
            f"ARXIV_OFFLINE is set and nothing is cached for {cleaned_topic!r}. "
            "Unset ARXIV_OFFLINE, or run this topic once online to warm the cache."
        )

    if client is None:
        client = make_client(settings)

    attempts = [q for q in (query, fallback_query) if q]
    papers: list[Paper] = []
    last_error: RetrievalError | None = None

    for index, attempt in enumerate(attempts):
        try:
            results = _fetch(
                client,
                attempt,
                settings.retrieval_max_results,
                simulate_throttle=using_real_network and settings.arxiv_force_429,
            )
        except RetrievalThrottled:
            if use_cache:
                stale = read_cache(settings, key, allow_stale=True)
                if stale:
                    logger.warning(
                        "arXiv throttled; serving %d stale cached papers instead", len(stale)
                    )
                    return stale
            raise
        except RetrievalError as exc:
            last_error = exc
            logger.warning("Search attempt %d failed: %s", index + 1, exc)
            continue

        papers = dedupe_by_paper_id([result_to_paper(r) for r in results])
        if papers:
            if index > 0:
                logger.info("Expanded query matched nothing; unquoted retry found %d", len(papers))
            if use_cache:
                write_cache(
                    settings,
                    key,
                    attempt,
                    papers,
                    max_results=settings.retrieval_max_results,
                )
            return papers

    if not attempts and last_error is None:
        # Both the assembled query and the plain fallback were blank, which only
        # happens for input that carries no searchable content.
        raise RetrievalError(
            f"No searchable terms could be derived from {cleaned_topic!r}. "
            "Try a longer or more specific topic."
        )
    if last_error is not None and not papers:
        # Every attempt errored rather than legitimately matching nothing.
        stale = read_cache(settings, key, allow_stale=True) if use_cache else None
        if stale:
            return stale
        raise last_error

    return papers


__all__ = [
    "QueryResolver",
    "RetrievalError",
    "RetrievalOffline",
    "RetrievalThrottled",
    "build_query",
    "cache_key",
    "dedupe_by_paper_id",
    "fetch_candidates",
    "make_client",
    "normalize_paper_id",
    "read_cache",
    "resolve_query",
    "result_to_paper",
    "to_paper",
    "version_suffix",
    "write_cache",
]
