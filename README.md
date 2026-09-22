# AI Research Landscape Agent

Enter an ML topic. Get a visual map of the research area instead of a paper list:
key papers, how they cluster, how they relate, where they contradict each other,
and what's still open.

Learning a new ML topic usually means dozens of papers and no starting point.
This gives you a structured entry point instead.

## How it works

A four-stage pipeline, with each stage visible while it runs:

| Stage | What happens | Why |
|---|---|---|
| **1. Retrieval** | `arxiv` searches for the topic and pulls candidates | arXiv handles retrieval only |
| **2. Reranking** | Cross-encoder + LLM judge score each paper by semantic relevance | A keyword match is not the best paper |
| **3. Extraction** | Per paper: problem, method, results, contribution, limitations | Structured reading beats raw abstracts |
| **4. Synthesis** | Clusters, typed relationships, tensions, open problems, reading path | The landscape, not the list |

Node positions are **computed, not decorative**: paper abstracts are embedded,
projected to 2D with UMAP, and clustered with HDBSCAN. The LLM names the clusters
and writes the prose. Clusters on screen therefore correspond to real structure
in the embedding space.

Full design rationale and the phase-by-phase build plan: **[`research-landscape-agent.md`](research-landscape-agent.md)**.

## Stack

**Backend** — Python 3.11+, FastAPI, `arxiv`, `sentence-transformers`, `umap-learn`, `scikit-learn`, SQLite (stdlib).
**Frontend** — Next.js (App Router), Tailwind CSS, `@xyflow/react`.
**LLM** — NVIDIA NIM (default) or OpenRouter, both via the OpenAI SDK.

## Setup

```bash
# 1. Backend environment
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r api/requirements.txt

# 2. Configure
cp .env.example .env
#    Set NVIDIA_API_KEY (or LLM_PROVIDER=openrouter + OPENROUTER_API_KEY).
#
#    Confirm the model slug against the live list rather than guessing it:
#      curl -s "$NVIDIA_BASE_URL/models" -H "Authorization: Bearer $NVIDIA_API_KEY"

# 3. Frontend dependencies
cd web && npm install && cd ..
```

## Running

Two processes, side by side:

```bash
make api    # FastAPI on http://localhost:8000
make web    # Next.js on http://localhost:3000
```

Then open http://localhost:3000 and search any ML topic in plain English —
`retrieval-augmented generation`, `diffusion policy learning`,
`quantization for LLMs`. The pipeline runs in the background while each stage
reports progress, then the landscape appears.

The first run downloads two models (~90 MB cross-encoder, ~440 MB embedder) into
`data/models/`. Subsequent runs are local-only.

## Headless use

Every stage is runnable without the UI, which is how prompt quality gets tuned:

```bash
.venv/bin/python scripts/run_pipeline.py --topic "retrieval-augmented generation" --stage retrieve
```

## Tests

```bash
cd api && ../.venv/bin/python -m pytest        # offline, no network
```

The suite never touches the network. arXiv has been returning 429/503 under its
documented rate limit since early 2026, so the retrieval tests drive the real
`arxiv.Client` against a captured response fixture (`api/tests/fixtures/arxiv_response.xml`)
instead of the live API.

## Notes and constraints

- **Abstracts only, for now.** Full-text PDF parsing is the next planned step.
- **arXiv rate limits are real and currently aggressive.** The client waits 3
  seconds between requests and caches every result set on disk. Do not lower
  `ARXIV_DELAY_SECONDS`.
- **PDFs are never served from this app.** arXiv's Terms of Use forbid
  redistributing e-prints, so papers link to their `arxiv.org/abs/` page.
