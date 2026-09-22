"""Pytest bootstrap.

Guarantees ``api/`` is importable (``import config``, ``import store``, ...)
regardless of which directory pytest was invoked from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

# Tests must never reach the network. The arXiv API has been returning 429/503
# under its documented rate limit since early 2026, so a test suite that depends
# on it is a test suite that fails for reasons unrelated to the code. Everything
# network-facing reads api/tests/fixtures/ instead.
os.environ.setdefault("ARXIV_OFFLINE", "1")

import pytest  # noqa: E402

from config import Settings  # noqa: E402
from models import Paper  # noqa: E402
from store import init_db, session  # noqa: E402


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture()
def conn(db_path):
    with session(db_path) as connection:
        yield connection


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="nim",
        nvidia_api_key="test-key",
        llm_model="test/model",
        db_path=tmp_path / "test.db",
        retrieval_cache_dir=tmp_path / "cache",
        arxiv_offline=True,
    )


def make_paper(paper_id: str = "2107.05580", version: str = "1", **overrides) -> Paper:
    base = {
        "paper_id": paper_id,
        "version": version,
        "title": f"Title for {paper_id}",
        "abstract": f"Abstract for {paper_id}.",
        "authors": ["A. Author"],
        "published": "2021-07-12T00:00:00Z",
        "updated": "2021-07-12T00:00:00Z",
        "primary_category": "cs.CL",
        "categories": ["cs.CL"],
        "abs_url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
    }
    base.update(overrides)
    return Paper(**base)
