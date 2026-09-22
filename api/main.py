"""FastAPI application: REST reads plus the SSE pipeline stream.

The stream is the interesting endpoint. Three details are deliberate:

* **It is a POST that returns ``text/event-stream``.** ``EventSource`` cannot
  send a body or set headers, so the frontend reads a streamed response body with
  ``fetch``. The cost is the browser's automatic reconnection, which is why every
  stage event is also persisted to the ``runs`` table and can be replayed.

* **``X-Accel-Buffering: no`` and ``Cache-Control: no-cache``.** Without them a
  reverse proxy buffers the whole response and every stage arrives at once, which
  looks exactly like a broken pipeline.

* **The pipeline runs as an async generator, not a background task.** Cancelling
  the HTTP request cancels the generator, so a closed browser tab stops the run
  instead of leaving it running to completion.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Path, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import service
import store
from config import ConfigError, Settings, load_settings
from llm.client import build_completer
from models import ExpandRequest, HealthOut, LandscapeDetail, LandscapeSummary, TopicRequest, utcnow
from pipeline.stages import PipelineRun

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Proxies (nginx, some serverless gateways) buffer text/event-stream by
    # default, which turns a live stream into one late burst.
    "X-Accel-Buffering": "no",
}


def sse_frame(event: str, data: dict[str, Any]) -> str:
    """Serialise one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _settings() -> Settings:
    try:
        return load_settings(require_llm=False)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings(require_llm=False)
    store.init_db(settings.db_path)
    settings.apply_model_cache_env()
    logger.info(
        "Ready. db=%s provider=%s model=%s llm_configured=%s",
        settings.db_path,
        settings.llm_provider,
        settings.llm_model,
        bool(settings.llm_api_key),
    )
    yield


app = FastAPI(
    title="AI Research Landscape Agent",
    description=(
        "Turns an ML topic into an interactive map: retrieval, reranking, "
        "structured extraction, and synthesis."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=load_settings(require_llm=False).cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Health and discovery
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    settings = _settings()
    return HealthOut(
        status="ok",
        llm_configured=bool(settings.llm_api_key and settings.llm_model),
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        prompt_version=settings.prompt_version,
    )


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """List model ids the configured provider offers.

    Exists so an ``LLM_MODEL`` value can be confirmed against the live catalogue
    instead of guessed from documentation or memory.
    """
    import httpx

    settings = _settings()
    if not settings.llm_api_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "No API key configured for LLM_PROVIDER="
                f"{settings.llm_provider}; cannot list models."
            ),
        )
    try:
        response = httpx.get(
            f"{settings.llm_base_url}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Model listing failed: {exc}") from exc
    payload = response.json()
    ids = sorted(
        item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)
    )
    return {"provider": settings.llm_provider, "count": len(ids), "models": ids}


# --------------------------------------------------------------------------- #
# Landscapes
# --------------------------------------------------------------------------- #


@app.get("/v1/landscapes", response_model=list[LandscapeSummary])
def list_landscapes() -> list[LandscapeSummary]:
    settings = _settings()
    with store.session(settings) as conn:
        return service.list_summaries(conn)


@app.get("/v1/landscapes/{landscape_id}", response_model=LandscapeDetail)
def get_landscape(landscape_id: int = Path(ge=1)) -> LandscapeDetail:
    settings = _settings()
    with store.session(settings) as conn:
        detail = service.build_detail(conn, landscape_id, settings)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No landscape with id {landscape_id}.")
    return detail


@app.get("/v1/landscapes/{landscape_id}/papers/{paper_id}")
def get_paper(landscape_id: int = Path(ge=1), paper_id: str = Path(min_length=1)) -> dict[str, Any]:
    settings = _settings()
    with store.session(settings) as conn:
        paper = service.fetch_paper_detail(conn, landscape_id, paper_id, settings)
    if paper is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paper {paper_id!r} is not part of landscape {landscape_id}.",
        )
    return paper.model_dump()


@app.delete("/v1/landscapes/{landscape_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_landscape(landscape_id: int = Path(ge=1)) -> Response:
    settings = _settings()
    with store.session(settings) as conn:
        if store.fetch_landscape(conn, landscape_id) is None:
            raise HTTPException(status_code=404, detail=f"No landscape with id {landscape_id}.")
        store.delete_landscape(conn, landscape_id)
    # Papers are shared across landscapes, so they are intentionally left behind.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/v1/landscapes/{landscape_id}/runs")
def get_runs(landscape_id: int = Path(ge=1)) -> dict[str, Any]:
    """Replay the persisted stage events, for a client that reconnected."""
    settings = _settings()
    with store.session(settings) as conn:
        if store.fetch_landscape(conn, landscape_id) is None:
            raise HTTPException(status_code=404, detail=f"No landscape with id {landscape_id}.")
        rows = store.fetch_runs(conn, landscape_id)
    return {"landscape_id": landscape_id, "runs": rows}


# --------------------------------------------------------------------------- #
# The pipeline stream
# --------------------------------------------------------------------------- #


def _stream_headers() -> dict[str, str]:
    return dict(SSE_HEADERS)


@app.post("/v1/landscapes/stream")
async def stream_landscape(request: TopicRequest = Body(...)) -> StreamingResponse:
    """Run the pipeline, streaming one event per stage transition."""
    settings = _settings()
    completer = build_completer(settings)
    run = PipelineRun(request.topic, settings, completer=completer)
    return StreamingResponse(
        _stream(run, settings),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )


@app.post("/v1/landscapes/{landscape_id}/expand")
async def expand_landscape(
    landscape_id: int = Path(ge=1), body: ExpandRequest | None = Body(default=None)
) -> StreamingResponse:
    """Grow an existing landscape, re-ranking the whole corpus."""
    settings = _settings()
    with store.session(settings) as conn:
        row = store.fetch_landscape(conn, landscape_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No landscape with id {landscape_id}.")

    topic = row.get("topic") or ""
    if not topic:
        raise HTTPException(status_code=409, detail="Landscape has no topic to expand.")

    completer = build_completer(settings)
    run = PipelineRun(
        topic, settings, completer=completer, expand_landscape_id=landscape_id
    )
    return StreamingResponse(
        _stream(run, settings, final_event="expanded"),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )


async def _stream(
    run: PipelineRun, settings: Settings, *, final_event: str = "done"
) -> AsyncIterator[str]:
    """Wrap a pipeline run as SSE frames, persisting each event for replay."""
    failed = False
    try:
        async for event in run.run():
            run_id = f"{event.run_id}:{event.stage}:{event.status}"
            if event.landscape_id is not None:
                _persist_event(run_id, event, settings)
            yield sse_frame("stage", event.model_dump())
            if event.status == "error":
                failed = True
                yield sse_frame(
                    "error",
                    {
                        "run_id": event.run_id,
                        "landscape_id": event.landscape_id,
                        "stage": event.stage,
                        "message": event.message,
                        "retryable": bool(event.payload.get("retryable")),
                    },
                )
                return
    except Exception as exc:  # noqa: BLE001 - the stream must close cleanly
        logger.exception("Stream failed")
        yield sse_frame(
            "error",
            {
                "run_id": run.state.run_id,
                "landscape_id": run.state.landscape_id,
                "stage": "retrieval",
                "message": str(exc) or exc.__class__.__name__,
                "retryable": False,
            },
        )
        return

    if failed:
        return
    yield sse_frame(
        final_event,
        {
            "run_id": run.state.run_id,
            "landscape_id": run.state.landscape_id,
            "url": f"/landscape/{run.state.landscape_id}",
            "papers": len(run.state.papers),
            "clusters": run.state.cluster_count,
        },
    )


def _persist_event(event_id: str, event: Any, settings: Settings) -> None:
    """Record a stage event so a reconnecting client can replay it."""
    try:
        with store.session(settings) as conn:
            store.insert_run(
                conn,
                {
                    "id": event_id,
                    "landscape_id": event.landscape_id,
                    "stage": event.stage,
                    "status": event.status,
                    "message": event.message,
                    "payload": event.payload,
                    "created_at": event.ts or utcnow(),
                },
            )
    except Exception:  # noqa: BLE001 - observability must never break the run
        logger.warning("Could not persist run event %s", event_id)


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


@app.get("/v1/stream-demo")
def stream_demo(topic: str = Query(min_length=2)) -> StreamingResponse:
    """GET variant that takes the topic from the query string.

    ``EventSource``-compatible, for a browser or a curl smoke test that wants the
    simplest possible thing that streams. The POST endpoint above is what the app
    itself uses.
    """
    settings = _settings()
    run = PipelineRun(topic, settings, completer=build_completer(settings))
    return StreamingResponse(
        _stream(run, settings), media_type="text/event-stream", headers=_stream_headers()
    )


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    settings = load_settings(require_llm=False)
    uvicorn.run(app, host="127.0.0.1", port=settings.api_port)
