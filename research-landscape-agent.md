# AI Research Landscape Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Tasks are ordered by the dependency graph and must be executed in sequence. Do not skip Phase 0.

**Goal:** Turn a plain-English ML topic into an interactive map of the research area — key papers, how they cluster, how they relate, where they contradict each other, and what's still open — built by a four-stage pipeline whose progress the user watches live.

**Architecture:** Two apps side by side. A FastAPI backend runs the pipeline as an SSE stream (`retrieval → rerank → extract → synthesize → layout`), persisting everything to a single SQLite file so a landscape can be expanded across runs. A Next.js frontend renders the resulting graph with React Flow, with 2D node positions computed server-side (embeddings → UMAP → HDBSCAN) so the map is spatially meaningful rather than decorative.

**Tech Stack:** Python 3.11+, FastAPI, `arxiv` 4.0.1, `sentence-transformers`, `umap-learn`, `scikit-learn`, stdlib `sqlite3`, Pydantic v2 · Next.js App Router, Tailwind, `@xyflow/react` · LLM via NVIDIA NIM (default) or OpenRouter, both through the OpenAI SDK.

---

## Phase 0: Documentation Discovery (do this first, do not skip)

Every API surface below was verified against live sources on 2026-09-22. **Read the cited sources before writing code.** The single largest source of failure in this project is an LLM writing arXiv code from memory, because the library removed its most-documented APIs in 2026.

### Allowed APIs — with sources

**`arxiv` 4.0.1** (source: `raw.githubusercontent.com/lukasschwab/arxiv.py/master/arxiv/__init__.py`)

| API | Signature / shape |
|---|---|
| `arxiv.Client` | `Client(page_size: int = 100, delay_seconds: float = 3.0, num_retries: int = 3)` |
| `Client.results` | `results(search: Search, offset: int = 0) -> Iterator[Result]` — **synchronous**, backed by `requests.Session` |
| `arxiv.Search` | `Search(query: str = "", id_list: list[str] | None = None, max_results: int | None = 100, sort_by: SortCriterion = Relevance, sort_order: SortOrder = Descending)` |
| `arxiv.SortCriterion` | `.Relevance`, `.LastUpdatedDate`, `.SubmittedDate` |
| `arxiv.SortOrder` | `.Ascending`, `.Descending` |
| `Result` fields | `entry_id, updated, published, title, authors: list[Result.Author], summary, comment, journal_ref, doi, primary_category, categories: list[str], links, pdf_url` |
| `Result.Author` | `.name`, `.affiliation: list[str]` |
| `Result` methods | `get_short_id()`, `source_url()`. `__eq__` compares `entry_id`. |
| Errors | `ArxivError`, `UnexpectedEmptyPageError`, `HTTPError`, `MissingFieldError` |

**arXiv Terms of Use** (source: `info.arxiv.org/help/api/tou.html`)
- One request every three seconds, single connection, across all machines you control.
- ✅ You may retrieve/store/share metadata, and store e-print content for personal or research use.
- ✅ Building a better search interface, a topic visualization, and a citation graph are all explicitly named as endorsed uses.
- ❌ **You must not store and serve arXiv e-prints (PDFs) from your own servers.** Link to `https://arxiv.org/abs/{id}`.

**arXiv operational reality** (source: arXiv API group thread `groups.google.com/a/arxiv.org/g/api/c/ycq8giRdZsQ`, Feb–Jun 2026)
- arXiv staff confirmed limits were tightened in Feb 2026; users reported persistent **429 and 503 while under the documented 1/3s limit**.
- The arxiv.py maintainer: **POST to `/api/query` bypasses the Fastly cache and 429s immediately; GET gets cache hits.** Use GET (arxiv.py already does).

**NVIDIA NIM** (source: `docs.nvidia.com/nim/large-language-models/latest/structured-generation.html`)
- OpenAI-compatible at `https://integrate.api.nvidia.com/v1`; chat via the OpenAI SDK with `base_url`.
- Structured generation is requested via **`extra_body={"nvext": {"guided_json": <json-schema>}}`**.

**OpenRouter** (source: `openrouter.ai/docs/guides/features/structured-outputs`)
- Base URL `https://openrouter.ai/api/v1`, API key as Bearer, OpenAI-compatible.
- Structured outputs via `response_format={"type": "json_schema", "json_schema": {...}}`, and you must also set `provider.require_parameters: true` to stop it routing to providers that ignore the schema.

**Clustering** (sources: `scikit-learn.org` `sklearn.cluster.HDBSCAN`; `umap-learn.readthedocs.io`)
- `sklearn.cluster.HDBSCAN(min_cluster_size=..., metric=...)` exists in scikit-learn ≥ 1.3 — **use it; do not add the standalone `hdbscan` package.**
- `umap.UMAP(..., random_state=42)` for reproducible coordinates. HDBSCAN labels noise as `-1`; `-1` is not a cluster.

**Reuse from the workspace:** `RAG-Pipeline-With-Hybrid-Search/retrieval/cross_encoder_reranker.py` (sigmoid calibration + fallback cap), `.../retrieval/llm_reranker.py` (0–10 judge rubric + strict alignment parsing), `.../config.py` (env settings dataclass), `.../api.py` (`StreamingResponse(..., media_type="text/event-stream")`), `.../generation/citations.py` (`lexical_support`, citation verification — the model for this project's evidence validator).

### Anti-pattern guards — what NOT to do

| ❌ Do not | Why |
|---|---|
| `arxiv.query(...)`, `arxiv.Search(...).results()`, `Result.download_pdf()`, `Result.download_source()`, `max_results=math.inf` | All removed in 4.0.0. Use `Client.results(search)`, `result.pdf_url`, `result.source_url()`, `max_results=None`. |
| POST to `export.arxiv.org/api/query` | Bypasses Fastly cache → immediate 429. |
| `response_format={"type":"json_schema"}` against NVIDIA NIM | NIM's documented path is `extra_body.nvext.guided_json`. |
| Trusting OpenRouter/any LLM to return valid JSON | Validate against the Pydantic schema and run a bounded repair loop on every call. |
| Min-max normalizing reranker scores within a batch | Forces the best candidate to 10 even when all are irrelevant — destroys absolute meaning. |
| `import hdbscan` | Deprecated path. Use `sklearn.cluster.HDBSCAN`. |
| Treating HDBSCAN label `-1` as a cluster | Noise is not a topic. Render it separately. |
| Hosting arXiv PDFs | ToU violation. Store `pdf_url` for linking only. |
| Calling `arxiv.Client` directly from an async route | It's synchronous; it will block the event loop. Wrap in `asyncio.to_thread`. |

---

## Architecture Decisions

| Decision | Rationale |
|---|---|
| **stdlib `sqlite3`, not an ORM** | Every table's DDL is written out in this plan, so there is nothing to invent. Adds zero dependencies and no Pydantic-version risk. WAL mode handles the single-writer-per-run pattern fine. |
| **`papers` keyed by version-stripped arXiv ID** | `Result.__eq__` compares `entry_id`, which *includes* `vN`. Without stripping, v1 and v3 of the same paper both land in the map. |
| **`paper_extractions` keyed by `(paper_id, prompt_version)`** | Prompt iteration is the main quality lever. This lets you re-extract everything without recomputing embeddings or disturbing the graph. |
| **`paper_embeddings` stores float32 BLOBs** | Re-running UMAP on a grown topic must never re-embed. |
| **Positions recomputed, never `umap.transform()`ed** | `transform()` places new points using an approximation that is inconsistent with the original embedding, so existing nodes visibly jump. Recompute with `random_state=42` and bump `landscapes.generation`. |
| **Cluster on the 2D coordinates, not the high-dim space** | Clusters then match the blobs the user actually sees. Tradeoff accepted deliberately. |
| **Typed edges (`extends`/`contradicts`/`applies`/`shares_method`)** | A generic "related" edge makes the "tensions" section meaningless. |
| **`evidence` field on every extraction** | Each extracted field must quote a verbatim span of the abstract, validated as a real substring. |
| **Two LLM backends behind one client** | NIM + OpenRouter differ in *how* you request JSON but not in the interface. `LLMClient.complete_json()` normalizes both. |
| **Extraction capped at top-N (60) papers** | Cost control. All ~200 candidates persist in `papers`; only the top-N go into `landscape_papers`. |

### Dependency graph

```
config.py + store.py (DDL)
        │
        ├── retrieve.py ──────► CLI: run_pipeline.py --stage retrieve
        │       │
        │       └── rerank.py ──► CLI: --stage rerank
        │               │
        │               └── extract.py ──► CLI: --stage extract
        │                       │
        │                       └── embed.py + cluster.py ──► CLI: --stage layout
        │                               │
        │                               └── synthesize.py ──► CLI: --stage synthesize
        │                                       │
        │                                       └── main.py (SSE + REST) ──► web/lib/types.ts
        │                                                                          │
        │                                                                          └── UI components
        └── tests/fixtures/arxiv_response.xml (offline; used by all tests)
```

Critical path: **Task 1 → 2 → 3 → 4 → 7a → 7b → 8 → 9 → 10.** Tasks 5 and 6 can run in parallel with 7a once Task 4 lands.

---

## Repository Layout

```
Research-agent/
├── research-landscape-agent.md      # this plan
├── README.md
├── .env.example
├── api/                             # FastAPI backend
│   ├── main.py                      # SSE + REST routes
│   ├── config.py                    # Settings dataclass from env
│   ├── models.py                    # Pydantic contract
│   ├── store.py                     # sqlite3 DDL + CRUD
│   ├── llm/{client,structured}.py   # NIM / OpenRouter + validate-and-repair
│   ├── pipeline/
│   │   ├── stages.py                # run_pipeline(): async generator of stage events
│   │   ├── retrieve.py              # arXiv + disk cache + version dedup
│   │   ├── rerank.py                # cross-encoder + LLM judge + blend
│   │   ├── extract.py               # structured extraction + evidence validation
│   │   ├── embed.py                 # sentence-transformers embeddings
│   │   ├── cluster.py               # UMAP 2D + HDBSCAN + noise handling
│   │   └── synthesize.py            # clusters, edges, tensions, open problems
│   ├── prompts/                     # versioned prompt templates
│   └── tests/
├── web/                             # Next.js app
├── scripts/run_pipeline.py          # headless CLI driver
└── data/                            # gitignored
```

---

## Data Model

`api/store.py` owns the DDL and applies it idempotently on startup. Tables: `topics`, `papers`, `paper_embeddings`, `landscapes`, `landscape_papers`, `clusters`, `edges`, `tensions`, `open_problems`, `reading_path`, `paper_extractions`, `runs`. Full DDL lives in `api/store.py`.

---

## SSE Event Protocol

```
event: stage
data: {"run_id":"r_8f2a","landscape_id":12,"stage":"retrieval","status":"running",
       "message":"Searching arXiv...","progress":{"current":0,"total":0},"payload":{},"ts":"..."}

event: progress   # intra-stage ticks; stage events always bracket them
event: done       # {"landscape_id":12,"url":"/landscape/12"}
event: error      # {"stage":"retrieval","message":"arXiv is throttling (429).","retryable":true}
```

Stages in fixed order: `retrieval → rerank → extraction → synthesis → layout → done`.

---

# Task List

## Task 1: Foundation — repo, settings, SQLite store, offline fixtures

**Files:** `api/config.py`, `api/store.py`, `api/models.py`, `api/requirements.txt`, `api/pyproject.toml`, `api/tests/test_store.py`, `api/tests/fixtures/arxiv_response.xml`, `.env.example`, `.gitignore`

**Acceptance criteria:** `pytest api/tests/test_store.py` passes with no network; `.env.example` documents every variable `Settings` reads; `data/` is gitignored.

## Task 2: arXiv retrieval — query builder, disk cache, dedup, CLI

**Files:** `api/pipeline/retrieve.py`, `api/prompts/query.py`, `api/tests/test_arxiv_query.py`, `api/tests/test_dedup.py`, `scripts/run_pipeline.py`

**Acceptance criteria:** tests pass offline; live run yields ≥100 unique papers; the second run issues no requests; a 429 surfaces as `RetrievalError` with a human-readable message.

**Do not** add retry/sleep logic of your own — `arxiv.Client` already enforces the delay.

## Task 3: Reranking — cross-encoder + LLM judge, blended

**Files:** `api/pipeline/rerank.py`, `api/prompts/rerank.py`, `api/tests/test_rerank_calibration.py`

**Acceptance criteria:** irrelevant candidates score low in isolation; every result carries `rerank_source`; the blend matches the configured weights; either backend failing degrades rather than throws.

## Task 4: Structured extraction with evidence validation

**Files:** `api/pipeline/extract.py`, `api/prompts/extract.py`, `api/llm/{client,structured}.py`, `api/tests/test_extract.py`

**Acceptance criteria:** fabricated quotes are rejected; a second run is fully cached; an LLM outage yields null extractions, not a failed run.

## Task 5: Embeddings, UMAP projection, HDBSCAN clusters

**Files:** `api/pipeline/embed.py`, `api/pipeline/cluster.py`, `api/tests/test_cluster.py`

**Acceptance criteria:** identical input yields identical coordinates; noise papers are a distinct group; `n < 5` doesn't crash; embeddings are computed once per paper per model.

## Task 6: Synthesis — clusters, relationships, tensions, open problems, reading path

**Files:** `api/pipeline/synthesize.py`, `api/prompts/{cluster,synthesize}.py`, `api/tests/test_synthesize.py`

**Acceptance criteria:** every edge and tension references real papers; edge kinds match the enum; a failed cluster-label call degrades to a placeholder.

## Task 7: FastAPI layer — persistence contracts, SSE orchestration, and growth

**Files:** `api/main.py`, `api/pipeline/stages.py`, `api/tests/test_api.py`

**Acceptance criteria:** the detail response matches the documented contract exactly; five stages stream in order; `expand` adds only new papers.

## Task 8: Next.js shell and the live stage timeline

**Files:** `web/lib/{sse,api,types}.ts`, `web/components/{SearchBar,StageTimeline}.tsx`, `web/app/{page,layout}.tsx`

**Acceptance criteria:** stages advance visibly during a real run; a backend error renders readably; a cancelled run aborts cleanly.

## Task 9: The reading map

**Files:** `web/components/{LandscapeCanvas,PaperNode,ClusterNode,PaperPanel,ClusterLegend,ReadingPath}.tsx`

**Acceptance criteria:** positions and clusters agree visually; every paper links to arXiv; the unclustered group is clearly separate; expand visibly grows the graph.

## Task 10: Verification phase

Offline suite green, three real topics end-to-end, growth checked, cost audited, anti-pattern greps clean, ToU compliance confirmed, README written.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| arXiv 429/503 (ongoing since Feb 2026) | High | GET only; disk cache; rely on `Client`'s delay; retries; offline fixture; readable throttling message |
| LLM returns unparseable JSON | High | validate + bounded repair; NIM `nvext.guided_json`, OpenRouter `json_schema` + `require_parameters` |
| Hallucinated extractions | High | mandatory verbatim `evidence` span per field |
| Cost blowup in extraction | Medium | cap at top 60; cache on `(paper_id, prompt_version, model)`; semaphore |
| CPU cross-encoder slow/OOM | Medium | `max_length=512`, `batch_size=16`, thread offload |
| UMAP/HDBSCAN instability | Medium | fixed `random_state=42`; recompute not `transform()`; noise handled separately |
| Blocking calls stalling SSE | Medium | `asyncio.to_thread` for arXiv, cross-encoder, UMAP |
| ToU violation by serving PDFs | High (legal) | `pdf_url` for linking only; abstracts-only in M1 |

## Open Questions

- **Default `LLM_MODEL`** — resolve from `GET /v1/models`; the current default matches the RAG-Pipeline value but should be confirmed.
- **`BAAI/bge-base-en-v1.5` vs SPECTER2** — the `EMBED_MODEL` knob exists for this.
- **Semantic Scholar citation edges** — deferred; `edges.kind` has room for a `cites` variant.
- **Auth** — none in M1; it is a local personal-agent tool.
