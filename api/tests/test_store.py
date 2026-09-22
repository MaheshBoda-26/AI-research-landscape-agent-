"""Store layer tests: DDL idempotency, cascades, and cache-key semantics."""

from __future__ import annotations

from conftest import make_paper
from models import PaperExtraction
from store import (
    count_embeddings,
    delete_landscape,
    existing_paper_ids,
    fetch_clusters,
    fetch_edges,
    fetch_embeddings,
    fetch_extraction,
    fetch_extractions,
    fetch_landscape_papers,
    fetch_landscape_row,
    fetch_open_problems,
    fetch_paper,
    fetch_reading_path,
    insert_landscape,
    insert_run,
    link_papers,
    list_landscapes,
    replace_clusters,
    replace_edges,
    replace_open_problems,
    replace_reading_path,
    replace_tensions,
    update_landscape_meta,
    update_paper_layout,
    upsert_embeddings,
    upsert_extraction,
    upsert_papers,
    upsert_topic,
)
from store import init_db as run_init_db
from models import RankedPaper


def _landscape(conn, topic: str = "retrieval-augmented generation") -> int:
    topic_id = upsert_topic(conn, topic)
    return insert_landscape(conn, topic_id, title="Test Landscape")


def _ranked(paper_id: str, rank: int = 1, score: float = 8.0) -> RankedPaper:
    return RankedPaper(
        paper=make_paper(paper_id), relevance_score=score, rank=rank, rerank_source="blend"
    )


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_init_db_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    run_init_db(db)
    run_init_db(db)  # must not raise
    from store import connect

    with connect(db) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "papers",
        "paper_embeddings",
        "landscapes",
        "landscape_papers",
        "clusters",
        "edges",
        "tensions",
        "open_problems",
        "reading_path",
        "paper_extractions",
        "runs",
        "topics",
    } <= tables


def test_foreign_keys_are_enforced(conn):
    # A landscape_papers row for a nonexistent landscape must be rejected.
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO landscape_papers (landscape_id, paper_id, rank, relevance_score, "
            "rerank_source, rationale, is_seed, added_at) VALUES (999, 'nope', 1, 1.0, 'x', '', 0, 'now')"
        )


def test_deleting_a_landscape_cascades(conn):
    lid = _landscape(conn)
    upsert_papers(conn, [make_paper("2107.05580")])
    link_papers(conn, lid, [_ranked("2107.05580")])
    replace_edges(
        conn,
        lid,
        [{"src_paper_id": "2107.05580", "dst_paper_id": "2107.05580", "kind": "extends"}],
    )
    replace_clusters(conn, lid, [{"local_label": 0, "label": "C", "paper_count": 1}])
    replace_reading_path(conn, lid, [{"paper_id": "2107.05580", "position": 1, "why": "start"}])

    assert delete_landscape(conn, lid) is True
    assert fetch_landscape_papers(conn, lid) == []
    assert fetch_edges(conn, lid) == []
    assert fetch_clusters(conn, lid) == []
    assert fetch_reading_path(conn, lid) == []
    # The paper itself is shared across landscapes and must survive.
    assert fetch_paper(conn, "2107.05580") is not None


# --------------------------------------------------------------------------- #
# Topics and papers
# --------------------------------------------------------------------------- #


def test_upsert_topic_is_case_and_whitespace_insensitive(conn):
    first = upsert_topic(conn, "Retrieval-Augmented   Generation")
    second = upsert_topic(conn, "  retrieval-augmented generation ")
    assert first == second


def test_upsert_papers_refreshes_metadata_without_duplicating(conn):
    upsert_papers(conn, [make_paper("2107.05580", version="1", title="Old title")])
    upsert_papers(conn, [make_paper("2107.05580", version="2", title="New title")])
    row = fetch_paper(conn, "2107.05580")
    assert row is not None
    assert row["title"] == "New title"
    assert row["version"] == "2"
    count = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
    assert count == 1


# --------------------------------------------------------------------------- #
# Ranked-paper links
# --------------------------------------------------------------------------- #


def test_link_papers_is_idempotent_and_preserves_first_rank(conn):
    """Growing a landscape must not reset a paper that is already on the map."""
    lid = _landscape(conn)
    upsert_papers(conn, [make_paper("a.0001"), make_paper("a.0002")])
    link_papers(conn, lid, [_ranked("a.0001", rank=1), _ranked("a.0002", rank=2)])
    first_added_at = {
        row["paper_id"]: row["rank"] for row in fetch_landscape_papers(conn, lid)
    }

    # Re-link both with different ranks: OR IGNORE must keep the original rows.
    link_papers(conn, lid, [_ranked("a.0001", rank=9), _ranked("a.0002", rank=8)])
    after = {row["paper_id"]: row["rank"] for row in fetch_landscape_papers(conn, lid)}
    assert after == first_added_at
    assert existing_paper_ids(conn, lid) == {"a.0001", "a.0002"}


def test_update_paper_layout_writes_cluster_and_coordinates(conn):
    lid = _landscape(conn)
    upsert_papers(conn, [make_paper("a.0001")])
    link_papers(conn, lid, [_ranked("a.0001")])
    update_paper_layout(conn, lid, [("a.0001", 3, 0.25, -0.5)])
    row = fetch_landscape_papers(conn, lid)[0]
    assert row["cluster_id"] == 3
    assert (row["x"], row["y"]) == (0.25, -0.5)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def test_embeddings_roundtrip_preserves_bytes(conn):
    upsert_papers(conn, [make_paper("a.0001")])
    vector = b"\x00\x01\x02\x03" * 4
    upsert_embeddings(conn, "test-model", [("a.0001", vector, 4)])
    assert fetch_embeddings(conn, "test-model")["a.0001"] == vector
    assert count_embeddings(conn, "test-model") == 1


def test_embeddings_are_keyed_by_model(conn):
    upsert_papers(conn, [make_paper("a.0001")])
    upsert_embeddings(conn, "model-a", [("a.0001", b"aaaa", 1)])
    upsert_embeddings(conn, "model-b", [("a.0001", b"bbbb", 1)])
    assert fetch_embeddings(conn, "model-a")["a.0001"] == b"aaaa"
    assert fetch_embeddings(conn, "model-b")["a.0001"] == b"bbbb"


# --------------------------------------------------------------------------- #
# Extractions
# --------------------------------------------------------------------------- #


def test_extraction_is_cached_per_prompt_version(conn):
    upsert_papers(conn, [make_paper("a.0001")])
    extraction = PaperExtraction(problem="P", method="M", confidence=0.8)
    upsert_extraction(conn, "a.0001", "extract_v1", "test/model", extraction)
    assert fetch_extraction(conn, "a.0001", "extract_v1") is not None
    # A different prompt version must be a cache miss, so prompt iteration
    # re-extracts without disturbing the v1 rows.
    assert fetch_extraction(conn, "a.0001", "extract_v2") is None


def test_extraction_upsert_replaces_same_version(conn):
    upsert_papers(conn, [make_paper("a.0001")])
    upsert_extraction(conn, "a.0001", "v1", "m", PaperExtraction(problem="old"))
    upsert_extraction(conn, "a.0001", "v1", "m", PaperExtraction(problem="new"))
    assert fetch_extraction(conn, "a.0001", "v1").problem == "new"


def test_fetch_extractions_bulk_returns_only_requested(conn):
    upsert_papers(conn, [make_paper("a.0001"), make_paper("a.0002")])
    upsert_extraction(conn, "a.0001", "v1", "m", PaperExtraction(problem="one"))
    upsert_extraction(conn, "a.0002", "v1", "m", PaperExtraction(problem="two"))
    got = fetch_extractions(conn, "v1", ["a.0001"])
    assert set(got) == {"a.0001"}


# --------------------------------------------------------------------------- #
# Landscape meta and listing
# --------------------------------------------------------------------------- #


def test_list_landscapes_counts_papers_and_real_clusters_only(conn):
    lid = _landscape(conn)
    upsert_papers(conn, [make_paper("a.0001"), make_paper("a.0002")])
    link_papers(conn, lid, [_ranked("a.0001", rank=1), _ranked("a.0002", rank=2)])
    replace_clusters(
        conn,
        lid,
        [
            {"local_label": 0, "label": "real", "paper_count": 1},
            {"local_label": -1, "label": "Unclustered", "paper_count": 1},
        ],
    )
    update_landscape_meta(conn, lid, title="RAG", summary="A summary")
    row = list_landscapes(conn)[0]
    assert row["paper_count"] == 2
    # -1 is noise, not a topic.
    assert row["cluster_count"] == 1
    assert row["topic"] == "retrieval-augmented generation"
    assert row["title"] == "RAG"


def test_open_problems_roundtrip_supporting_ids(conn):
    lid = _landscape(conn)
    replace_open_problems(
        conn,
        lid,
        [{"statement": "S", "why_open": "W", "supporting_paper_ids": ["a.0001", "a.0002"]}],
    )
    problems = fetch_open_problems(conn, lid)
    assert problems[0]["supporting_paper_ids"] == ["a.0001", "a.0002"]


def test_tensions_replace_is_not_append(conn):
    lid = _landscape(conn)
    replace_tensions(conn, lid, [{"statement": "first"}])
    replace_tensions(conn, lid, [{"statement": "second"}])
    rows = conn.execute("SELECT statement FROM tensions WHERE landscape_id = ?", (lid,)).fetchall()
    assert [r["statement"] for r in rows] == ["second"]


def test_run_log_upserts_by_id(conn):
    lid = _landscape(conn)
    insert_run(conn, "r1", lid, "retrieval", "running")
    insert_run(conn, "r1", lid, "retrieval", "done", payload={"count": 42})
    row = conn.execute("SELECT * FROM runs WHERE id = 'r1'").fetchone()
    assert row["status"] == "done"


def test_landscape_row_exposes_topic_and_params(conn):
    topic_id = upsert_topic(conn, "diffusion policy learning")
    lid = insert_landscape(conn, topic_id, title="DPL", params={"k": 1})
    row = fetch_landscape_row(conn, lid)
    assert row["topic"] == "diffusion policy learning"
    assert row["params"] == {"k": 1}
