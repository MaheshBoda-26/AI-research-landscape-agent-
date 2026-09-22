"""Shared test fixtures.

Every fixture is hermetic: the database, the arXiv response cache, and the model
cache all live under ``tmp_path``. No test in this suite may touch the network.

Paths are injected by ``dataclasses.replace`` on a frozen ``Settings``, which is
why ``Settings`` is a frozen dataclass rather than a mutable object.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:  # pragma: no cover - defensive
    sys.path.insert(0, str(API_ROOT))

import store  # noqa: E402
from config import Settings  # noqa: E402
from models import Paper, PaperExtraction  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings rooted at tmp_path, with a freshly created schema."""
    base = Settings.from_env()
    isolated = replace(
        base,
        db_path=tmp_path / "landscapes.db",
        retrieval_cache_dir=tmp_path / "cache" / "arxiv",
        arxiv_offline=True,
        # Tests never call a real model. ``llm_api_key`` is a derived property,
        # not a field, so it is cleared by clearing the two real fields.
        nvidia_api_key="",
        openrouter_api_key="",
    )
    isolated.ensure_dirs()
    store.init_db(isolated.db_path)
    return isolated


@pytest.fixture
def conn(settings: Settings):
    """A live connection to the isolated database."""
    connection = store.connect(settings.db_path)
    try:
        yield connection
    finally:
        connection.close()


def make_paper(
    paper_id: str = "2107.05580",
    *,
    version: str = "1",
    title: str = "Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
    abstract: str = "We introduce a retrieval-augmented generation model that combines a "
    "dense retriever with a sequence-to-sequence generator.",
    **overrides: object,
) -> Paper:
    """Build a Paper with sensible defaults for tests."""
    data: dict[str, object] = {
        "paper_id": paper_id,
        "version": version,
        "title": title,
        "abstract": abstract,
        "authors": ["A. Researcher", "B. Scientist"],
        "published": "2021-07-12T00:00:00Z",
        "updated": "2021-07-12T00:00:00Z",
        "primary_category": "cs.CL",
        "categories": ["cs.CL", "cs.LG"],
        "abs_url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
    }
    data.update(overrides)
    return Paper(**data)  # type: ignore[arg-type]


def make_extraction(**overrides: object) -> PaperExtraction:
    """Build an extraction whose evidence quotes are real substrings of the
    default abstract produced by ``make_paper``."""
    data: dict[str, object] = {
        "problem": "Knowledge-intensive tasks need external evidence.",
        "method": "A dense retriever plus a seq2seq generator.",
        "results": "Improved generation on knowledge-intensive benchmarks.",
        "contribution": "A retrieval-augmented generation model.",
        "limitations": "Not reported in the abstract.",
        "datasets": ["Natural Questions"],
        "metrics": ["exact match"],
        "novelty": "substantial",
        "evidence": {"method": "combines a dense retriever with a sequence-to-sequence generator"},
        "confidence": 0.8,
    }
    data.update(overrides)
    return PaperExtraction(**data)  # type: ignore[arg-type]


@pytest.fixture
def paper() -> Paper:
    return make_paper()


@pytest.fixture
def arxiv_fixture_path() -> Path:
    return FIXTURES / "arxiv_response.xml"
