"""Version-dedup and result-normalization tests.

Version handling is the subtle part: ``arxiv.Result.__eq__`` compares ``entry_id``,
which includes the ``vN`` suffix, so v1 and v3 of one paper are *different*
results. Without collapsing them the map shows the same paper twice.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fake_arxiv import mk_result
from pipeline.retrieve import dedupe_by_paper_id, to_paper

EARLY = datetime(2021, 1, 1, tzinfo=timezone.utc)
LATE = datetime(2022, 5, 5, tzinfo=timezone.utc)


def test_same_paper_two_versions_collapses_to_newest():
    results = [
        mk_result("http://arxiv.org/abs/2107.05580v1", updated=EARLY),
        mk_result("http://arxiv.org/abs/2107.05580v3", updated=LATE),
    ]
    papers = dedupe_by_paper_id(results)
    assert len(papers) == 1
    assert papers[0].paper_id == "2107.05580"
    assert papers[0].version == "3"


def test_newest_version_wins_even_when_it_appears_first():
    """arXiv does not guarantee version ordering within a result set."""
    results = [
        mk_result("http://arxiv.org/abs/2107.05580v3", updated=LATE),
        mk_result("http://arxiv.org/abs/2107.05580v1", updated=EARLY),
    ]
    papers = dedupe_by_paper_id(results)
    assert len(papers) == 1 and papers[0].version == "3"


def test_dedup_preserves_first_appearance_order():
    results = [
        mk_result("http://arxiv.org/abs/1111.1111v1"),
        mk_result("http://arxiv.org/abs/2222.2222v1"),
        mk_result("http://arxiv.org/abs/1111.1111v2", updated=LATE),
    ]
    papers = dedupe_by_paper_id(results)
    assert [p.paper_id for p in papers] == ["1111.1111", "2222.2222"]


def test_distinct_papers_are_kept():
    results = [
        mk_result("http://arxiv.org/abs/1111.1111v1"),
        mk_result("http://arxiv.org/abs/2222.2222v1"),
    ]
    assert len(dedupe_by_paper_id(results)) == 2


def test_unversioned_result_keeps_empty_version():
    papers = dedupe_by_paper_id([mk_result("http://arxiv.org/abs/quant-ph/0201082v1")])
    assert papers[0].paper_id == "quant-ph/0201082"


def test_to_paper_splits_version_and_normalizes_abs_url():
    paper = to_paper(mk_result("http://arxiv.org/abs/2411.18583v2"))
    assert paper.paper_id == "2411.18583"
    assert paper.version == "2"
    # The feed's <id> is plain http; users should get the https abstract page.
    assert paper.abs_url == "https://arxiv.org/abs/2411.18583"


def test_to_paper_collapses_whitespace_in_title_and_abstract():
    """arXiv titles arrive with newlines and authors with padded affiliations."""
    paper = to_paper(
        mk_result(
            title="  A Very\n   Long Title  ",
            summary="Line one.\nLine two.\n\n  Line three.  ",
        )
    )
    assert paper.title == "A Very Long Title"
    assert paper.abstract == "Line one. Line two. Line three."


def test_to_paper_handles_missing_optional_fields():
    paper = to_paper(mk_result(doi="", journal_ref=""))
    assert paper.doi == ""
    assert paper.journal_ref == ""
    assert paper.pdf_url == ""
