"""Retrieval tests that run entirely offline.

These drive the real ``arxiv.Client`` — URL formatting, pagination, retry logic,
and Atom parsing — with only the HTTP session swapped for a fake. The response
body is a real arXiv response captured once and committed as a fixture.
"""

from __future__ import annotations

import json

import pytest
from fake_arxiv import fake_client, fixture_bytes

from config import Settings
from pipeline.retrieve import (
    RetrievalError,
    cache_key,
    fetch_candidates,
    read_cache,
    write_cache,
)


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "nvidia_api_key": "test-key",
        "llm_model": "test/model",
        "db_path": tmp_path / "test.db",
        "retrieval_cache_dir": tmp_path / "cache",
        "retrieval_max_results": 5,
        "arxiv_offline": False,
    }
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_fetch_candidates_parses_the_fixture(tmp_path):
    settings = _settings(tmp_path)
    papers = fetch_candidates(
        "retrieval-augmented generation", settings, client=fake_client(), use_cache=False
    )
    assert len(papers) == 5
    first = papers[0]
    assert first.paper_id == "2411.18583"
    assert first.version == "1"
    assert first.title.startswith("Automated Literature Review")
    assert first.abs_url == "https://arxiv.org/abs/2411.18583"
    assert first.pdf_url == "https://arxiv.org/pdf/2411.18583v1"
    assert "cs.CL" in first.categories
    assert first.primary_category == "cs.CL"
    assert first.authors  # names were parsed


def test_fetch_candidates_requests_the_expanded_query(tmp_path):
    settings = _settings(tmp_path)
    client = fake_client()
    fetch_candidates("RAG", settings, client=client, use_cache=False)
    assert len(client._session.calls) == 1
    assert "export.arxiv.org/api/query" in client._session.calls[0]
    assert "search_query=" in client._session.calls[0]


def test_fetch_candidates_dedupes_across_pages(tmp_path):
    """The fake session always serves the same page, so asking for more results
    than it contains forces pagination and exercises version collapsing."""
    settings = _settings(tmp_path, retrieval_max_results=10)
    papers = fetch_candidates(
        "retrieval-augmented generation", settings, client=fake_client(), use_cache=False
    )
    assert len(papers) == 5  # not 10: the duplicate page collapsed
    assert len({p.paper_id for p in papers}) == 5


def test_fetch_candidates_returns_empty_when_the_query_matches_nothing(tmp_path):
    empty_feed = b"<?xml version='1.0' encoding='UTF-8'?><feed xmlns='http://www.w3.org/2005/Atom'></feed>"
    settings = _settings(tmp_path)
    papers = fetch_candidates("nothing at all", settings, client=fake_client(empty_feed), use_cache=False)
    assert papers == []


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_fetch_candidates_writes_and_reuses_the_cache(tmp_path):
    settings = _settings(tmp_path)
    first_client = fake_client()
    fetch_candidates("retrieval-augmented generation", settings, client=first_client)
    assert len(first_client._session.calls) == 1

    # Second call with a client that would explode if used: must hit the cache.
    class ExplodingClient:
        def results(self, *_args, **_kwargs):
            raise AssertionError("network was touched despite a warm cache")

    papers = fetch_candidates("retrieval-augmented generation", settings, client=ExplodingClient())
    assert len(papers) == 5


def test_cache_roundtrip_preserves_papers(tmp_path):
    settings = _settings(tmp_path)
    papers = fetch_candidates(
        "retrieval-augmented generation", settings, client=fake_client(), use_cache=False
    )
    write_cache(settings, "abc123", "all:test", papers)
    restored = read_cache(settings, "abc123")
    assert restored is not None
    assert [p.paper_id for p in restored] == [p.paper_id for p in papers]
    assert restored[0].abstract == papers[0].abstract


def test_expired_cache_is_ignored(tmp_path):
    settings = _settings(tmp_path, retrieval_cache_ttl_hours=0)
    # ttl=0 disables expiry, so a fresh entry is returned...
    papers = fetch_candidates("t", settings, client=fake_client(), use_cache=False)
    write_cache(settings, "k", "q", papers)
    assert read_cache(settings, "k") is not None

    expired = _settings(tmp_path, retrieval_cache_ttl_hours=1)
    import os
    import time

    path = expired.retrieval_cache_dir / "k.json"
    old = time.time() - 7200
    os.utime(path, (old, old))
    assert read_cache(expired, "k") is None


def test_corrupt_cache_entry_is_ignored_not_fatal(tmp_path):
    settings = _settings(tmp_path)
    settings.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    (settings.retrieval_cache_dir / "broken.json").write_text("{not json")
    assert read_cache(settings, "broken") is None


def test_cache_key_is_stable_and_query_specific():
    assert cache_key("all:rag", 200) == cache_key("all:rag", 200)
    assert cache_key("all:rag", 200) != cache_key("all:rag", 100)
    assert cache_key("all:rag", 200) != cache_key("all:dpo", 200)


def test_cache_file_records_the_query(tmp_path):
    settings = _settings(tmp_path)
    papers = fetch_candidates("t", settings, client=fake_client(), use_cache=False)
    key = cache_key('all:"t"', 5)
    write_cache(settings, key, 'all:"t"', papers)
    payload = json.loads((settings.retrieval_cache_dir / f"{key}.json").read_text())
    assert payload["query"] == 'all:"t"'
    assert payload["max_results"] == len(papers)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


def test_offline_without_cache_raises_a_readable_error(tmp_path):
    settings = _settings(tmp_path, arxiv_offline=True)
    with pytest.raises(RetrievalError) as exc:
        fetch_candidates("unseen topic", settings)
    assert "ARXIV_OFFLINE" in str(exc.value)


def test_429_surfaces_a_throttling_message_not_a_traceback(tmp_path):
    """arXiv has been rate-limiting aggressively since early 2026; the user needs
    to be told that, not handed a library exception."""
    settings = _settings(tmp_path)
    with pytest.raises(RetrievalError) as exc:
        fetch_candidates("rag", settings, client=fake_client(status_code=429), use_cache=False)
    message = str(exc.value)
    assert "429" in message
    assert "throttl" in message.lower()


def test_503_surfaces_as_a_server_side_error(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(RetrievalError) as exc:
        fetch_candidates("rag", settings, client=fake_client(status_code=503), use_cache=False)
    assert "503" in str(exc.value)


def test_empty_topic_query_is_rejected(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(RetrievalError):
        fetch_candidates("   ", settings, client=fake_client(), use_cache=False)


def test_fixture_is_a_real_arxiv_feed():
    """Guard against the fixture being replaced with a hand-written stub."""
    body = fixture_bytes()
    assert b"<feed" in body
    assert b"export.arxiv.org" in body or b"arxiv.org/api" in body
    assert b"<opensearch:totalResults>" in body
