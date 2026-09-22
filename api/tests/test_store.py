"""Tests for the SQLite persistence layer.

These assert the modelling decisions that the rest of the pipeline depends on:
the schema is idempotent, deleting a landscape cascades, papers are keyed by a
version-stripped id, and the ``(paper_id, prompt_version)`` extraction cache
actually returns a hit.
"""

from __future__ import annotations

import sqlite3

import pytest

import store
from conftest import make_extraction, make_paper
from models import EdgeClaim, OpenProblem, ReadingStep, Tension

EXPECTED_TABLES = {
    "clusters",
    "edges",
    "landscape_papers",
    "landscapes",
    "open_problems",
    "paper_embeddings",
    "paper_extractions",
    "papers",
    "reading_path",
    "runs",
    "tensions",
    "topics",
}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_init_db_creates_every_table(conn):
    assert EXPECTED_TABLES <= _tables(conn)


def test_init_db_is_idempotent(settings):
    store.init_db(settings.db_path)
    store.init_db(settings.db_path)  # must not raise
    connection = store.connect(settings.db_path)
    try:
        assert EXPECTED_TABLES <= _tables(connection)
    finally:
        connection.close()


def test_wal_mode_is_enabled(conn):
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_foreign_keys_are_enforced(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO landscape_papers (landscape_id, paper_id, rank, relevance_score, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (9999, "does-not-exist", 1, 1.0, "2026-01-01T00:00:00Z"),
        )


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #


def test_topic_upsert_is_case_and_space_insensitive(conn):
    first = store.upsert_topic(conn, "Retrieval-Augmented Generation")
    second = store.upsert_topic(conn, "  retrieval-augmented   generation ")
    assert first == second
    assert store.fetch_topic(conn, first)["query_text"] == "Retrieval-Augmented Generation"


def test_distinct_topics_get_distinct_ids(conn):
    a = store.upsert_topic(conn, "diffusion policy learning")
    b = store.upsert_topic(conn, "quantization for LLMs")
    assert a != b


# --------------------------------------------------------------------------- #
# Papers
# --------------------------------------------------------------------------- #


def test_upsert_papers_reports_only_new_rows(conn):
    paper = make_paper()
    assert store.upsert_papers(conn, [paper]) == 1
    assert store.upsert_papers(conn, [paper]) == 0
    assert store.count_papers(conn) == 1


def test_paper_ids_are_version_stripped_by_construction(conn):
    """v1 and v3 of one paper must occupy a single row, not two."""
    store.upsert_papers(conn, [make_paper(paper_id="2107.05580", version="1", title="old")])
    store.upsert_papers(conn, [make_paper(paper_id="2107.05580", version="3", title="new")])
    assert store.count_papers(conn) == 1
    stored = store.fetch_paper(conn, "2107.05580")
    assert stored is not None and stored.version == "3" and stored.title == "new"


def test_paper_roundtrips_authors_and_categories(conn):
    store.upsert_papers(conn, [make_paper()])
    stored = store.fetch_paper(conn, "2107.05580")
    assert stored is not None
    assert stored.authors == ["A. Researcher", "B. Scientist"]
    assert stored.categories == ["cs.CL", "cs.LG"]
    assert stored.abs_link == "https://arxiv.org/abs/2107.05580"


def test_abs_link_falls_back_when_missing(conn):
    store.upsert_papers(conn, [make_paper(abs_url="")])
    stored = store.fetch_paper(conn, "2107.05580")
    assert stored is not None and stored.abs_link == "https://arxiv.org/abs/2107.05580"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def test_embeddings_roundtrip_as_float32_bytes(conn):
    import numpy as np

    store.upsert_papers(conn, [make_paper()])
    vector = np.array([0.5, -1.25, 3.0], dtype=np.float32)
    store.upsert_embeddings(conn, "test-model", {"2107.05580": vector.tobytes()})

    fetched = store.fetch_embeddings(conn, "test-model")
    restored = np.frombuffer(fetched["2107.05580"], dtype=np.float32)
    assert np.allclose(restored, vector)


def test_embeddings_are_scoped_by_model(conn):
    store.upsert_papers(conn, [make_paper()])
    store.upsert_embeddings(conn, "model-a", {"2107.05580": b"\x00" * 8})
    assert store.fetch_embeddings(conn, "model-b") == {}


# --------------------------------------------------------------------------- #
# Extractions — the prompt-version cache
# --------------------------------------------------------------------------- #


def test_extraction_is_cached_per_prompt_version(conn):
    store.upsert_papers(conn, [make_paper()])
    store.upsert_extraction(conn, "2107.05580", "extract_v1", make_extraction(), model="m")

    hit = store.fetch_extraction(conn, "2107.05580", "extract_v1")
    assert hit is not None and hit.problem is not None
    # Bumping the prompt version must MISS, so a re-extraction is forced.
    assert store.fetch_extraction(conn, "2107.05580", "extract_v2") is None


def test_extraction_upsert_overwrites_in_place(conn):
    store.upsert_papers(conn, [make_paper()])
    store.upsert_extraction(conn, "2107.05580", "extract_v1", make_extraction(problem="first"))
    store.upsert_extraction(conn, "2107.05580", "extract_v1", make_extraction(problem="second"))
    hit = store.fetch_extraction(conn, "2107.05580", "extract_v1")
    assert hit is not None and hit.problem == "second"
    count = conn.execute("SELECT COUNT(*) AS n FROM paper_extractions").fetchone()["n"]
    assert count == 1


def test_extraction_roundtrips_lists_and_evidence(conn):
    store.upsert_papers(conn, [make_paper()])
    store.upsert_extraction(conn, "2107.05580", "extract_v1", make_extraction())
    hit = store.fetch_extraction(conn, "2107.05580", "extract_v1")
    assert hit is not None
    assert hit.datasets == ["Natural Questions"]
    assert hit.novelty == "substantial"
    assert hit.evidence["method"].startswith("combines a dense retriever")
    assert hit.confidence == pytest.approx(0.8)


def test_null_extraction_survives_a_roundtrip(conn):
    """A failed LLM call stores a nulled extraction rather than crashing."""
    from models import PaperExtraction

    store.upsert_papers(conn, [make_paper()])
    store.upsert_extraction(conn, "2107.05580", "extract_v1", PaperExtraction())
    hit = store.fetch_extraction(conn, "2107.05580", "extract_v1")
    assert hit is not None and hit.problem is None and hit.novelty == "unclear"


# --------------------------------------------------------------------------- #
# Landscapes
# --------------------------------------------------------------------------- #


@pytest.fixture
def seeded(conn):
    """A landscape with two papers, clusters, and every synthesis artifact."""
    topic_id = store.upsert_topic(conn, "retrieval-augmented generation")
    store.upsert_papers(
        conn, [make_paper("2107.05580", title="A"), make_paper("2108.00001", title="B")]
    )
    landscape_id = store.insert_landscape(conn, topic_id=topic_id, title="RAG")
    for rank, pid in enumerate(["2107.05580", "2108.00001"], start=1):
        store.link_paper(
            conn,
            landscape_id,
            paper_id=pid,
            rank=rank,
            relevance_score=9.0 - rank,
            rerank_source="blend",
        )
    return landscape_id


def test_insert_and_fetch_landscape(conn):
    topic_id = store.upsert_topic(conn, "RAG")
    landscape_id = store.insert_landscape(conn, topic_id=topic_id, title="RAG landscape")
    row = store.fetch_landscape(conn, landscape_id)
    assert row is not None
    assert row["status"] == "running" and row["generation"] == 1 and row["topic"] == "RAG"


def test_deleting_a_landscape_cascades(conn, seeded):
    assert len(store.fetch_landscape_papers(conn, seeded)) == 2
    store.delete_landscape(conn, seeded)
    assert store.fetch_landscape_papers(conn, seeded) == []
    assert store.fetch_clusters(conn, seeded) == []
    # The shared papers row must survive: it is reused by other landscapes.
    assert store.count_papers(conn) == 2


def test_link_paper_is_idempotent_and_keeps_position(conn, seeded):
    store.update_layout(conn, seeded, {"2107.05580": (0.25, -0.5, 3)})
    store.link_paper(
        conn, seeded, paper_id="2107.05580", rank=1, relevance_score=9.5, rerank_source="llm"
    )
    papers = store.fetch_landscape_papers(conn, seeded)
    target = next(p for p in papers if p["paper_id"] == "2107.05580")
    assert len(papers) == 2
    assert target["relevance_score"] == pytest.approx(9.5)
    assert target["rerank_source"] == "llm"
    # Re-ranking must not wipe the layout computed by a previous run.
    assert target["x"] == pytest.approx(0.25)
    assert target["cluster_id"] == 3


def test_landscape_paper_ids_and_layout(conn, seeded):
    assert store.landscape_paper_ids(conn, seeded) == {"2107.05580", "2108.00001"}
    store.update_layout(conn, seeded, {"2107.05580": (0.1, 0.2, -1)})
    target = next(p for p in store.fetch_landscape_papers(conn, seeded) if p["paper_id"] == "2107.05580")
    assert (target["x"], target["y"], target["cluster_id"]) == (0.1, 0.2, -1)


def test_bump_generation_increments(conn, seeded):
    assert store.bump_generation(conn, seeded) == 2
    assert store.bump_generation(conn, seeded) == 3


def test_list_landscapes_reports_counts_and_excludes_unclustered(conn, seeded):
    store.replace_clusters(
        conn,
        seeded,
        [
            {"local_label": 0, "label": "Dense retrieval", "paper_count": 1},
            {"local_label": -1, "label": "Unclustered", "paper_count": 1},
        ],
    )
    rows = store.list_landscapes(conn)
    assert len(rows) == 1
    assert rows[0]["paper_count"] == 2
    # -1 is noise, not a topic, so it must not inflate the cluster count.
    assert rows[0]["cluster_count"] == 1


def test_update_landscape_sets_status_summary_and_timestamp(conn, seeded):
    store.update_landscape(conn, seeded, title="New", summary="S", status="ready")
    row = store.fetch_landscape(conn, seeded)
    assert row is not None
    assert (row["title"], row["summary"], row["status"]) == ("New", "S", "ready")


def test_find_landscape_for_topic_prefers_newest_generation(conn, seeded):
    topic_id = store.upsert_topic(conn, "retrieval-augmented generation")
    second = store.insert_landscape(conn, topic_id=topic_id, title="RAG v2")
    store.bump_generation(conn, second)
    found = store.find_landscape_for_topic(conn, topic_id)
    assert found is not None and found["id"] == second


# --------------------------------------------------------------------------- #
# Synthesis artifacts
# --------------------------------------------------------------------------- #


def test_replace_semantics_for_edges_tensions_and_problems(conn, seeded):
    first = EdgeClaim(
        src_paper_id="2107.05580", dst_paper_id="2108.00001", kind="extends", rationale="r1"
    )
    second = EdgeClaim(
        src_paper_id="2108.00001", dst_paper_id="2107.05580", kind="contradicts", rationale="r2"
    )
    store.replace_edges(conn, seeded, [first])
    assert len(store.fetch_edges(conn, seeded)) == 1
    store.replace_edges(conn, seeded, [first, second])
    assert len(store.fetch_edges(conn, seeded)) == 2
    # Duplicate (src, dst, kind) collapses via the unique index.
    store.replace_edges(conn, seeded, [first, first])
    assert len(store.fetch_edges(conn, seeded)) == 1

    store.replace_tensions(conn, seeded, [Tension(statement="they disagree", paper_a_id="a", paper_b_id="b")])
    assert store.fetch_tensions(conn, seeded)[0]["statement"] == "they disagree"

    store.replace_open_problems(
        conn,
        seeded,
        [OpenProblem(statement="still open", why_open="no benchmark", supporting_paper_ids=["2107.05580"])],
    )
    problems = store.fetch_open_problems(conn, seeded)
    assert problems[0]["supporting_paper_ids"] == ["2107.05580"]


def test_reading_path_is_ordered_and_joined_to_titles(conn, seeded):
    store.replace_reading_path(
        conn,
        seeded,
        [
            ReadingStep(paper_id="2108.00001", position=2, why="later"),
            ReadingStep(paper_id="2107.05580", position=1, why="start here"),
        ],
    )
    steps = store.fetch_reading_path(conn, seeded)
    assert [s["position"] for s in steps] == [1, 2]
    assert steps[0]["why"] == "start here"
    assert steps[0]["title"] == "A"


def test_cluster_labels_apply_onto_existing_rows(conn, seeded):
    from models import ClusterLabel

    store.replace_clusters(conn, seeded, [{"local_label": 0, "label": "", "paper_count": 2}])
    store.replace_cluster_labels(
        conn,
        seeded,
        [ClusterLabel(local_label=0, label="Dense retrieval", description="retriever-centric")],
    )
    clusters = store.fetch_clusters(conn, seeded)
    assert clusters[0]["label"] == "Dense retrieval"
    assert clusters[0]["description"] == "retriever-centric"


def test_cluster_id_by_label_maps_local_to_row_id(conn, seeded):
    store.replace_clusters(conn, seeded, [{"local_label": -1}, {"local_label": 0}])
    mapping = store.cluster_id_by_label(conn, seeded)
    assert set(mapping) == {-1, 0}
    assert mapping[0] != mapping[-1]


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def test_insert_and_fetch_run_events(conn, seeded):
    store.insert_run(
        conn,
        {
            "id": "run-1",
            "landscape_id": seeded,
            "stage": "retrieval",
            "status": "done",
            "message": "found 200 papers",
            "payload": {"count": 200},
        },
    )
    rows = store.fetch_runs(conn, seeded)
    assert len(rows) == 1
    assert rows[0]["stage"] == "retrieval"
    import json

    assert json.loads(rows[0]["payload_json"]) == {"count": 200}


def test_run_insert_is_replace_by_id(conn, seeded):
    base = {"id": "run-2", "landscape_id": seeded, "stage": "rerank", "status": "running"}
    store.insert_run(conn, {**base, "message": "starting"})
    store.insert_run(conn, {**base, "status": "done", "message": "finished"})
    rows = store.fetch_runs(conn, seeded)
    assert len(rows) == 1 and rows[0]["status"] == "done"
