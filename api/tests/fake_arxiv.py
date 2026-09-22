"""Offline stand-ins for the arXiv API.

The arXiv API has been returning 429/503 under its documented rate limit since
early 2026, so no test may depend on it. These fakes let the retrieval tests
exercise the *real* ``arxiv.Client`` code path — URL formatting, pagination,
retry logic, and Atom parsing — by swapping only the HTTP session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from arxiv import Result

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "arxiv_response.xml"


def fixture_bytes() -> bytes:
    return FIXTURE_PATH.read_bytes()


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
        self.text = content.decode("utf-8", errors="replace")


class FakeSession:
    """Mimics ``requests.Session`` just enough for ``arxiv.Client``.

    Records every URL requested so tests can assert on pagination and caching.
    """

    def __init__(self, content: bytes | None = None, status_code: int = 200) -> None:
        self.content = content if content is not None else fixture_bytes()
        self.status_code = status_code
        self.calls: list[str] = []
        self.headers: list[dict | None] = []

    def get(self, url: str, headers: dict | None = None, **_: object) -> FakeResponse:
        self.calls.append(url)
        self.headers.append(headers)
        return FakeResponse(self.content, self.status_code)


def fake_client(content: bytes | None = None, status_code: int = 200):
    """An ``arxiv.Client`` with a fake session and no throttling delays."""
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=0.0, num_retries=0)
    client._session = FakeSession(content, status_code)  # test-only session swap
    return client


def mk_result(
    entry_id: str = "http://arxiv.org/abs/2107.05580v1",
    *,
    updated: datetime | None = None,
    title: str = "A title",
    summary: str = "An abstract.",
    categories: list[str] | None = None,
    primary_category: str = "cs.CL",
    doi: str = "",
    journal_ref: str = "",
) -> Result:
    """Build an ``arxiv.Result`` without touching the network."""
    moment = updated or datetime(2021, 7, 12, tzinfo=timezone.utc)
    return Result(
        entry_id=entry_id,
        updated=moment,
        published=moment,
        title=title,
        authors=[],
        summary=summary,
        comment="",
        journal_ref=journal_ref,
        doi=doi,
        primary_category=primary_category,
        categories=categories or ["cs.CL"],
        links=[],
    )
