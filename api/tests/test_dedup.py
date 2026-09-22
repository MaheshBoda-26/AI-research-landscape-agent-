"""Tests for identifier normalization, dedup, caching, and throttling.

All offline: the captured Atom response in ``tests/fixtures/`` is parsed with the
same ``arxiv._feed`` parser that ``arxiv.Client`` uses internally. Reaching into
the private module is the only way to exercise real parse output without
touching the network, and the network is exactly what must stay out of the test
suite given arXiv's current rate-limit behavior.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from arxiv import _feed

from config import Settings
from conftest import make_paper
from models import Paper
from pipeline.retrieve import (
    RetrievalOffline,
    RetrievalThrottled,
    cache_key,
    dedupe_by_paper_id,
    fetch_candidates,
    normalize_paper_id,
    read_cache,
    result_to_paper,
    version_suffix,
    write_cache,
)
from prompts.query import build_query, heuristic_query, plain_query


@pytest.fixture
def fixture_results(arxiv_fixture_path: Path):
    """Real ``arxiv.Result`` objects parsed from the committed capture."""
    if not arxiv_fixture_path.exists():  # pragma: no cover - fixture is committed
        pytest.skip("arxiv_response.xml fixture is missing")
    return list(_feed.parse(arxiv_fixture_path.read_bytes()).results)


# --------------------------------------------------------------------------- #
# Identifier handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("short_id", "expected"),
    [
        ("2411.18583v1", "2411.18583"),
        ("2411.18583v12", "2411.18583"),
        ("2411.18583", "2411.18583"),
        ("quant-ph/0201082v1", "quant-ph/0201082"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("", ""),
    ],
)
def test_normalize_paper_id(short_id, expected):
    assert normalize_paper_id(short_id) == expected


@pytest.mark.parametrize(
    ("short_id", "expected"),
    [("2411.18583v1", "1"), ("2411.18583v10", "10"), ("2411.18583", ""), ("", "")],
)
def test_version_suffix(short_id, expected):
    assert version_suffix(short_id) == expected


def test_version_is_not_confused_by_a_v_inside_the_id():
    """`v` appears in old-style archive names; only a trailing marker counts."""
    assert normalize_paper_id("solv-int/9701001v2") == "solv-int/9701001"
    assert version_suffix("solv-int/9701001v2") == "2"


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #


def test_two_versions_of_one_paper_collapse_to_the_newest():
    result = dedupe_by_paper_id(
        [
            make_paper(paper_id="2107.05580", version="1", title="old"),
            make_paper(paper_id="2107.05580", version="3", title="new"),
        ]
    )
    assert len(result) == 1
    assert result[0].version == "3" and result[0].title == "new"


def test_dedup_keeps_the_higher_version_regardless_of_arrival_order():
    result = dedupe_by_paper_id(
        [
            make_paper(paper_id="a", version="10", title="v10"),
            make_paper(paper_id="a", version="9", title="v9"),
        ]
    )
    assert result[0].title == "v10"


def test_dedup_preserves_relevance_order_of_first_appearance():
    result = dedupe_by_paper_id(
        [
            make_paper(paper_id="b", version="1"),
            make_paper(paper_id="a", version="1"),
            make_paper(paper_id="b", version="2"),
        ]
    )
    assert [p.paper_id for p in result] == ["b", "a"]


def test_dedup_skips_papers_without_an_id():
    assert dedupe_by_paper_id([make_paper(paper_id=""), make_paper(paper_id="a")]) == [
        make_paper(paper_id="a")
    ]


def test_dedup_of_an_unversioned_duplicate_keeps_the_first():
    result = dedupe_by_paper_id([make_paper(paper_id="a", title="first"), make_paper(paper_id="a", title="second")])
    assert len(result) == 1 and result[0].title == "first"


# --------------------------------------------------------------------------- #
# Result conversion, against the real captured response
# --------------------------------------------------------------------------- #


def test_fixture_parses_into_expected_number_of_results(fixture_results):
    """Five entries, deliberately fewer than the requested max.

    The fake session always serves the same page, so a fixture smaller than the
    requested limit is what forces ``Client`` to paginate and makes the
    cross-page version collapsing testable.
    """
    assert len(fixture_results) == 5


def test_every_fixture_result_converts_to_a_usable_paper(fixture_results):
    papers = [result_to_paper(r) for r in fixture_results]
    for paper in papers:
        assert paper.paper_id
        assert paper.title and paper.abstract
        assert paper.abs_url == f"https://arxiv.org/abs/{paper.paper_id}"
        assert paper.primary_category
        assert paper.authors


def test_converted_ids_never_carry_a_version_suffix(fixture_results):
    versioned = 0
    for result in fixture_results:
        paper = result_to_paper(result)
        # The raw short id may carry a version; the stored id never does.
        if version_suffix(result.get_short_id()):
            versioned += 1
        # ``v(\d+)$`` must not fire on an id whose last character merely happens
        # to be a digit, which is the common case for new-style ids.
        assert version_suffix(paper.paper_id) == ""
    # Guard the guard: if no fixture entry were versioned, this test would be
    # vacuously green and would stop protecting the version-stripping rule.
    assert versioned > 0


def test_converted_papers_dedupe_cleanly(fixture_results):
    papers = [result_to_paper(r) for r in fixture_results]
    deduped = dedupe_by_paper_id(papers)
    assert len(deduped) == len({p.paper_id for p in papers})


def test_published_is_an_iso_string_not_a_datetime(fixture_results):
    paper = result_to_paper(fixture_results[0])
    assert isinstance(paper.published, str)
    assert paper.published.startswith("20")
    datetime.fromisoformat(paper.published.replace("Z", "+00:00"))


def test_abstracts_are_long_enough_to_justify_a_512_token_window(fixture_results):
    """Guards the CROSS_ENCODER_MAX_LENGTH=512 decision.

    Roughly four characters per token, so a 256-token window truncates around
    1000 characters — and the tail of an abstract is where the results and
    contribution usually live. At least one fixture abstract must exceed that,
    or the 512 setting is unjustified.
    """
    lengths = sorted(len(result_to_paper(r).abstract) for r in fixture_results)
    assert max(lengths) > 1000, f"no abstract long enough to matter: {lengths}"


def test_none_journal_ref_and_doi_become_empty_strings(fixture_results):
    paper = result_to_paper(fixture_results[0])
    assert isinstance(paper.journal_ref, str) and isinstance(paper.doi, str)


def test_rerank_text_combines_title_and_abstract():
    paper = make_paper(title="T", abstract="A")
    assert paper.rerank_text == "T\n\nA"


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #


def _key_for(topic: str, settings: Settings) -> str:
    query = build_query(heuristic_query(topic))
    return cache_key(f"{query}||{plain_query(topic)}", settings.retrieval_max_results)


def _age_cache_file(settings: Settings, key: str, *, hours: float) -> Path:
    """Backdate a cache file.

    Freshness is decided by mtime, so aging the file is the honest way to test
    expiry — it cannot disagree with the filesystem the way a rewritten
    ``created_at`` field could.
    """
    path = settings.retrieval_cache_dir / f"{key}.json"
    old = time.time() - hours * 3600
    os.utime(path, (old, old))
    return path


def test_cache_roundtrip(settings: Settings):
    papers = [make_paper(paper_id="a"), make_paper(paper_id="b")]
    key = _key_for("some topic", settings)
    write_cache(settings, key, "all:q", papers)
    restored = read_cache(settings, key)
    assert restored is not None
    assert [p.paper_id for p in restored] == ["a", "b"]
    assert restored[0].title == papers[0].title


def test_cache_miss_returns_none(settings: Settings):
    assert read_cache(settings, "nonexistent-key") is None


def test_cache_respects_ttl_but_serves_stale_when_asked(settings: Settings):
    key = _key_for("topic", settings)
    write_cache(settings, key, "q", [make_paper()])
    _age_cache_file(settings, key, hours=99)

    assert read_cache(settings, key) is None
    assert read_cache(settings, key, allow_stale=True) is not None


def test_a_zero_ttl_disables_expiry(settings: Settings):
    """Used by tests and demos that need a cache to stay warm indefinitely."""
    from dataclasses import replace

    no_expiry = replace(settings, retrieval_cache_ttl_hours=0)
    key = _key_for("topic", settings)
    write_cache(no_expiry, key, "q", [make_paper()])
    _age_cache_file(no_expiry, key, hours=999)
    assert read_cache(no_expiry, key) is not None


def test_corrupt_cache_is_ignored_rather_than_raising(settings: Settings):
    key = _key_for("topic", settings)
    settings.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    (settings.retrieval_cache_dir / f"{key}.json").write_text("{not json")
    assert read_cache(settings, key) is None
    assert read_cache(settings, key, allow_stale=True) is None


def test_cache_skips_individual_malformed_papers(settings: Settings):
    key = _key_for("topic", settings)
    settings.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    # No created_at needed: freshness is decided by mtime, not by a field that
    # could drift out of sync with the file itself.
    (settings.retrieval_cache_dir / f"{key}.json").write_text(
        json.dumps({"papers": [make_paper().model_dump(), {"paper_id": "broken"}]})
    )
    restored = read_cache(settings, key)
    assert restored is not None and len(restored) == 1


# --------------------------------------------------------------------------- #
# fetch_candidates — offline and throttled
# --------------------------------------------------------------------------- #


def test_offline_without_cache_names_the_topic(settings: Settings):
    with pytest.raises(RetrievalOffline) as excinfo:
        fetch_candidates("quantum error correction for ML", settings)
    assert "quantum error correction for ML" in str(excinfo.value)


def test_offline_serves_a_warm_cache(settings: Settings):
    topic = "sparse attention"
    key = _key_for(topic, settings)
    write_cache(settings, key, "q", [make_paper(paper_id="x")])
    papers = fetch_candidates(topic, settings)
    assert [p.paper_id for p in papers] == ["x"]


def test_offline_serves_an_expired_cache_deliberately(settings: Settings):
    """Offline mode is the 'arXiv unreachable' case, so stale data wins.

    Refusing to serve an expired cache here would make the offline hook useless
    for any cache older than the TTL, which is the exact situation it exists to
    handle.
    """
    topic = "sparse attention"
    key = _key_for(topic, settings)
    write_cache(settings, key, "q", [make_paper(paper_id="stale")])
    _age_cache_file(settings, key, hours=99)
    papers = fetch_candidates(topic, settings)
    assert [p.paper_id for p in papers] == ["stale"]


def test_force_429_raises_throttled_when_nothing_is_cached(settings: Settings):
    from dataclasses import replace

    throttled = replace(settings, arxiv_offline=False, arxiv_force_429=True)
    with pytest.raises(RetrievalThrottled) as excinfo:
        fetch_candidates("a topic with no cache", throttled)
    assert "429" in str(excinfo.value) or "throttl" in str(excinfo.value)


def test_force_429_falls_back_to_stale_cache(settings: Settings):
    """Throttling is the common case; stale results beat a failed run."""
    from dataclasses import replace

    topic = "throttled topic"
    key = _key_for(topic, settings)
    write_cache(settings, key, "q", [make_paper(paper_id="stale")])
    _age_cache_file(settings, key, hours=99)

    throttled = replace(settings, arxiv_offline=False, arxiv_force_429=True)
    papers = fetch_candidates(topic, throttled)
    assert [p.paper_id for p in papers] == ["stale"]


@pytest.mark.live
def test_live_fetch_returns_a_real_corpus(settings: Settings):
    """Opt in with: RUN_LIVE_TESTS=1 python -m pytest -m live"""
    from dataclasses import replace

    live = replace(settings, arxiv_offline=False, retrieval_max_results=40)
    papers = fetch_candidates("retrieval-augmented generation", live)
    assert len(papers) >= 30
    assert all(isinstance(p, Paper) and p.abstract for p in papers)
    assert len({p.paper_id for p in papers}) == len(papers)
