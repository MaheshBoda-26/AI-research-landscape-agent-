"""Tests for the HTTP layer: the response contract and the SSE stream.

The stream test runs the **real** orchestration — real stage sequence, real
persistence, real degradation paths — with only the five leaves that touch the
network or a model patched. Patching ``PipelineRun`` itself would have tested the
SSE framing and nothing else, which is the wrong half.

No LLM key is configured, so extraction, cluster naming, and synthesis all take
their no-LLM branches. That is deliberate: it proves the pipeline completes and
produces a viewable landscape even with no model available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
import store
from conftest import make_paper
from models import Paper, PaperExtraction, RankedPaper
from pipeline import cluster as cluster_mod
from pipeline import retrieve as retrieve_mod

TOPIC = "retrieval-augmented generation"

PAPERS = [
    make_paper(
        paper_id=f"2406.0000{i}",
        title=f"Retrieval-Augmented Generation Paper {i}",
        abstract=(
            "We introduce a retrieval-augmented generation model that combines a dense "
            "retriever with a sequence-to-sequence generator. On HotpotQA it improves "
            f"exact match by {i} points. The approach is not evaluated on open-ended "
            "generation."
        ),
    )
    for i in range(12)
]


def fake_ranked(papers: list[Paper]) -> list[RankedPaper]:
    ranked = []
    for index, paper in enumerate(papers):
        logit = 9.0 - index * 0.2
        ranked.append(
            RankedPaper(
                paper=paper,
                relevance_score=9.99,
                cross_encoder_score=9.99,
                cross_encoder_logit=logit,
                relative_score=10.0 - index * 0.5,
                rerank_source="cross-encoder",
                rank=index + 1,
            )
        )
    return ranked


def fake_extractions(papers: list[Paper]) -> dict[str, PaperExtraction]:
    out = {}
    for paper in papers:
        # Sliced out of the paper's own abstract, so the quote is genuinely
        # verbatim however the caller ordered the papers.
        marker = "improves exact match by"
        start = paper.abstract.index(marker)
        quote = paper.abstract[start : paper.abstract.index(" points", start) + len(" points")]
        out[paper.paper_id] = PaperExtraction(
            problem="Grounding generation in retrieved evidence.",
            method="Dense retrieval plus a seq2seq generator.",
            results="Exact match improves on HotpotQA.",
            contribution="A retrieval-augmented generation model.",
            limitations=None,
            datasets=["HotpotQA"],
            metrics=["exact match"],
            novelty="substantial",
            evidence={"results": quote},
            confidence=0.85,
        )
    return out


def fake_layout(matrix: np.ndarray, settings) -> cluster_mod.Layout:
    count = matrix.shape[0]
    coords = np.stack(
        [np.linspace(-1.0, 1.0, count), np.sin(np.arange(count))], axis=1
    ).astype(np.float32)
    # Two clear clusters plus one outlier, so the -1 handling is exercised.
    labels = np.array([0] * (count - 1) + [-1], dtype=int)
    return cluster_mod.Layout(coords=coords, labels=labels)


@pytest.fixture
def patched(monkeypatch):
    """Replace the five leaves that need the network or a model."""
    monkeypatch.setattr(retrieve_mod, "fetch_candidates", lambda *a, **k: list(PAPERS))
    monkeypatch.setattr(
        "pipeline.stages.rerank_mod.rank_papers", lambda topic, papers, s, c=None: fake_ranked(papers)
    )

    def fake_extract_all(papers, completer, settings, conn, on_progress=None):
        """Stand in for the LLM call, but persist exactly like the real one.

        Writing through ``store.upsert_extraction`` is what makes the
        read-back assertions meaningful: the API serves extractions out of the
        database, not out of pipeline memory.
        """
        out = fake_extractions(list(papers))
        if on_progress:
            on_progress(len(papers), len(papers))
        if conn is not None:
            for paper_id, extraction in out.items():
                store.upsert_extraction(
                    conn, paper_id, settings.prompt_version, extraction, model="test"
                )
            conn.commit()
        return out

    monkeypatch.setattr("pipeline.stages.extract_mod.extract_all", fake_extract_all)

    def fake_embed(papers, settings, conn=None, embedder=None):
        return {
            paper.paper_id: np.random.default_rng(i).normal(size=(8,)).astype(np.float32)
            for i, paper in enumerate(papers)
        }

    monkeypatch.setattr("pipeline.stages.embed_mod.embed_papers", fake_embed)
    monkeypatch.setattr("pipeline.stages.cluster_mod.layout", fake_layout)
    yield monkeypatch


@pytest.fixture
def client(tmp_path: Path, patched):
    """A TestClient whose database lives in tmp_path.

    Env vars are set through the same ``monkeypatch`` instance the leaf patches
    use, so both are undone together.
    """
    monkeypatch = patched
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("RETRIEVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ARXIV_OFFLINE", "1")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(main.app) as test_client:
        yield test_client


def parse_frames(body: str) -> list[tuple[str, dict]]:
    frames = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data: dict = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:") :].strip())
        frames.append((event, data))
    return frames


def run_stream(client) -> list[tuple[str, dict]]:
    response = client.post("/v1/landscapes/stream", json={"topic": TOPIC})
    assert response.status_code == 200
    return parse_frames(response.text)


# --------------------------------------------------------------------------- #
# Health and listing
# --------------------------------------------------------------------------- #


def test_health_reports_no_llm_configured(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is False


def test_landscape_list_starts_empty(client):
    assert client.get("/v1/landscapes").json() == []


def test_short_topic_is_rejected(client):
    assert client.post("/v1/landscapes/stream", json={"topic": "x"}).status_code == 422


def test_whitespace_topic_is_rejected(client):
    assert client.post("/v1/landscapes/stream", json={"topic": "   "}).status_code == 422


def test_unknown_landscape_is_404(client):
    assert client.get("/v1/landscapes/9999").status_code == 404


def test_unknown_paper_is_404(client):
    assert client.get("/v1/landscapes/9999/papers/nope").status_code == 404


# --------------------------------------------------------------------------- #
# The stream
# --------------------------------------------------------------------------- #


def test_stream_emits_every_stage_in_order(client):
    """The signal that the UI timeline depends on."""
    frames = run_stream(client)
    stage_frames = [data for event, data in frames if event == "stage"]
    running = [f["stage"] for f in stage_frames if f["status"] == "running"]
    # `running` may repeat per progress tick; the first occurrence of each stage
    # must follow the declared order exactly.
    first_seen = list(dict.fromkeys(running))
    assert first_seen == ["retrieval", "rerank", "extraction", "layout", "synthesis"]


def test_every_stage_reports_done(client):
    frames = run_stream(client)
    done = {
        data["stage"] for event, data in frames if event == "stage" and data["status"] == "done"
    }
    assert done == {"retrieval", "rerank", "extraction", "layout", "synthesis"}


def test_stream_ends_with_a_done_frame(client):
    frames = run_stream(client)
    assert frames[-1][0] == "done"
    assert frames[-1][1]["url"].startswith("/landscape/")


def test_stage_events_carry_progress_and_landscape_id(client):
    frames = [data for event, data in run_stream(client) if event == "stage"]
    extraction = [f for f in frames if f["stage"] == "extraction"]
    assert any(f["progress"]["total"] == len(PAPERS) for f in extraction)
    # Once retrieval has persisted the landscape, later events carry its id.
    synthesis = [f for f in frames if f["stage"] == "synthesis"]
    assert all(f["landscape_id"] is not None for f in synthesis)


def test_retrieval_payload_reports_the_paper_count(client):
    frames = [data for event, data in run_stream(client) if event == "stage"]
    retrieval_done = next(
        f for f in frames if f["stage"] == "retrieval" and f["status"] == "done"
    )
    assert retrieval_done["payload"]["count"] == len(PAPERS)


def test_the_landscape_is_ready_after_the_stream(client):
    run_stream(client)
    summaries = client.get("/v1/landscapes").json()
    assert len(summaries) == 1
    assert summaries[0]["status"] == "ready"
    assert summaries[0]["paper_count"] == len(PAPERS)


def test_a_throttled_arxiv_produces_an_error_frame_not_a_crash(client, monkeypatch):
    def throttled(*_args, **_kwargs):
        raise retrieve_mod.RetrievalThrottled("arXiv is throttling this client (HTTP 429).")

    monkeypatch.setattr(retrieve_mod, "fetch_candidates", throttled)
    frames = run_stream(client)

    errors = [data for event, data in frames if event == "error"]
    assert len(errors) == 1
    assert errors[0]["retryable"] is True
    assert "429" in errors[0]["message"]
    # A throttled run yields no ``done`` frame, so the client does not navigate.
    assert not any(event == "done" for event, _ in frames)


def test_a_failed_retrieval_creates_no_landscape(client, monkeypatch):
    def boom(*_args, **_kwargs):
        raise retrieve_mod.RetrievalError("arXiv is unreachable")

    monkeypatch.setattr(retrieve_mod, "fetch_candidates", boom)
    frames = run_stream(client)
    assert any(event == "error" for event, _ in frames)
    # Nothing was created: retrieval failed before the landscape existed.
    assert client.get("/v1/landscapes").json() == []


def test_an_empty_corpus_is_reported_plainly(client, monkeypatch):
    monkeypatch.setattr(retrieve_mod, "fetch_candidates", lambda *a, **k: [])
    frames = run_stream(client)
    errors = [data for event, data in frames if event == "error"]
    assert errors and "no papers" in errors[0]["message"]


# --------------------------------------------------------------------------- #
# Detail contract
# --------------------------------------------------------------------------- #


@pytest.fixture
def detail(client) -> dict:
    run_stream(client)
    landscape_id = client.get("/v1/landscapes").json()[0]["id"]
    return client.get(f"/v1/landscapes/{landscape_id}").json()


def test_detail_exposes_every_documented_field(detail):
    required = {
        "id",
        "topic",
        "title",
        "summary",
        "status",
        "generation",
        "clusters",
        "papers",
        "edges",
        "tensions",
        "open_problems",
        "reading_path",
    }
    assert required <= set(detail)


def test_every_paper_has_the_fields_the_canvas_needs(detail):
    required = {
        "paper_id",
        "title",
        "abstract",
        "abs_url",
        "rank",
        "relevance_score",
        "cross_encoder_logit",
        "relative_score",
        "rerank_source",
        "cluster_id",
        "x",
        "y",
        "extraction",
    }
    for paper in detail["papers"]:
        assert required <= set(paper)


def test_papers_are_ordered_by_rank(detail):
    ranks = [paper["rank"] for paper in detail["papers"]]
    assert ranks == sorted(ranks)


def test_paper_cluster_ids_resolve_to_a_real_cluster(detail):
    """A dangling cluster_id would render as an orphaned node."""
    cluster_ids = {cluster["id"] for cluster in detail["clusters"]}
    for paper in detail["papers"]:
        if paper["cluster_id"] is not None:
            assert paper["cluster_id"] in cluster_ids


def test_the_unclustered_bucket_is_flagged_and_named_plainly(detail):
    unclustered = [c for c in detail["clusters"] if c["is_unclustered"]]
    assert len(unclustered) == 1
    assert unclustered[0]["label"] == "Unclustered"
    # The synthetic layout puts exactly one paper in the noise bucket.
    noise = [p for p in detail["papers"] if p["cluster_id"] == unclustered[0]["id"]]
    assert len(noise) == 1


def test_real_clusters_are_not_treated_as_noise(detail):
    real = [c for c in detail["clusters"] if not c["is_unclustered"]]
    assert len(real) == 1


def test_unclustered_never_inflates_the_cluster_count(client, detail):
    summary = client.get("/v1/landscapes").json()[0]
    assert summary["cluster_count"] == len(
        [c for c in detail["clusters"] if not c["is_unclustered"]]
    )


def test_extractions_are_attached_to_papers(detail):
    with_extraction = [p for p in detail["papers"] if p["extraction"]]
    assert with_extraction
    assert with_extraction[0]["extraction"]["datasets"] == ["HotpotQA"]


def test_scores_are_exposed_on_both_scales(detail):
    paper = detail["papers"][0]
    assert paper["relevance_score"] > 9.0  # absolute, saturating
    assert paper["cross_encoder_logit"] is not None  # resolves the ordering
    assert paper["relative_score"] == pytest.approx(10.0)  # best in this landscape


def test_positions_are_normalized_into_unit_range(detail):
    xs = [abs(paper["x"]) for paper in detail["papers"]]
    ys = [abs(paper["y"]) for paper in detail["papers"]]
    assert max(xs) <= 1.0 + 1e-6
    assert max(ys) <= 1.0 + 1e-6


def test_without_an_llm_the_summary_says_synthesis_did_not_run(detail):
    assert "not produced" in detail["summary"]
    assert detail["edges"] == []


def test_single_paper_endpoint_returns_one_paper(client, detail):
    paper_id = detail["papers"][0]["paper_id"]
    landscape_id = detail["id"]
    body = client.get(f"/v1/landscapes/{landscape_id}/papers/{paper_id}").json()
    assert body["paper_id"] == paper_id


def test_paper_from_another_landscape_is_404(client, detail):
    landscape_id = detail["id"]
    assert (
        client.get(f"/v1/landscapes/{landscape_id}/papers/9999.99999").status_code == 404
    )


# --------------------------------------------------------------------------- #
# Runs, expansion, deletion
# --------------------------------------------------------------------------- #


def test_run_events_are_persisted_for_replay(client, detail):
    body = client.get(f"/v1/landscapes/{detail['id']}/runs").json()
    stages = {row["stage"] for row in body["runs"]}
    assert {"retrieval", "rerank", "extraction", "layout", "synthesis"} <= stages


def test_expanding_grows_the_generation_and_keeps_papers(client, detail):
    landscape_id = detail["id"]
    frames = parse_frames(client.post(f"/v1/landscapes/{landscape_id}/expand", json={}).text)
    assert frames[-1][0] == "expanded"

    after = client.get(f"/v1/landscapes/{landscape_id}").json()
    assert after["generation"] >= 2
    assert len(after["papers"]) == len(detail["papers"])
    assert {p["paper_id"] for p in after["papers"]} == {
        p["paper_id"] for p in detail["papers"]
    }


def test_expand_of_an_unknown_landscape_is_404(client):
    assert client.post("/v1/landscapes/9999/expand", json={}).status_code == 404


def test_delete_removes_the_landscape_but_keeps_papers(client, detail):
    landscape_id = detail["id"]
    assert client.delete(f"/v1/landscapes/{landscape_id}").status_code == 204
    assert client.get(f"/v1/landscapes/{landscape_id}").status_code == 404
    assert client.get("/v1/landscapes").json() == []
    # Papers are shared across landscapes and must survive.
    with store.session(main._settings()) as conn:
        assert store.count_papers(conn) == len(PAPERS)


def test_delete_of_an_unknown_landscape_is_404(client):
    assert client.delete("/v1/landscapes/9999").status_code == 404


# --------------------------------------------------------------------------- #
# SSE framing
# --------------------------------------------------------------------------- #


def test_sse_frame_shape():
    frame = main.sse_frame("stage", {"a": 1})
    assert frame == 'event: stage\ndata: {"a": 1}\n\n'


def test_sse_frame_terminates_with_a_blank_line():
    assert main.sse_frame("done", {}).endswith("\n\n")


def test_sse_frame_serialises_unusual_types():
    from datetime import datetime

    assert "2026" in main.sse_frame("x", {"when": datetime(2026, 1, 1)})


def test_stream_response_headers_disable_buffering(client):
    response = client.post("/v1/landscapes/stream", json={"topic": TOPIC})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
