"""SQLite persistence layer.

Deliberately stdlib ``sqlite3`` and not an ORM: the schema is small, every table
is auditable, and there is nothing to invent. WAL mode is enabled so a streaming
pipeline run can keep writing stage events while the API reads a previous
landscape.

Key modelling decisions:

* ``papers`` is keyed by the **version-stripped** arXiv id and is shared across
  every landscape. It is the dedup boundary: ``arxiv.Result.__eq__`` compares
  ``entry_id``, which includes the ``vN`` suffix, so v1 and v3 of the same paper
  are distinct upstream results and would both land in the map without stripping.
* ``paper_extractions`` is keyed by ``(paper_id, prompt_version)`` so that
  iterating on the extraction prompt never forces a re-embed or a re-layout.
* ``paper_embeddings`` stores float32 vectors as BLOBs so a grown topic can be
  re-projected without re-embedding a single paper.
* ``reading_path`` is not in the original sketch of the schema; it is added here
  because the synthesized reading order has to survive a server restart, and
  ``LandscapeDetail.reading_path`` is part of the API contract.

Every function takes an explicit connection. Callers use the ``session()``
context manager, which owns commit/rollback and closes the connection. Opening a
fresh connection per unit of work is intentional: it is cheap under WAL and it
keeps connections from crossing thread boundaries when blocking work is pushed
through ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, Settings, load_settings
from models import (
    UNCLUSTERED_LABEL,
    ClusterLabel,
    EdgeClaim,
    OpenProblem,
    Paper,
    PaperExtraction,
    ReadingStep,
    Tension,
    utcnow,
)

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS topics (
  id INTEGER PRIMARY KEY,
  query_text TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_query ON topics(lower(trim(query_text)));

CREATE TABLE IF NOT EXISTS papers (
  paper_id     TEXT PRIMARY KEY,
  version      TEXT NOT NULL DEFAULT '',
  title        TEXT NOT NULL,
  abstract     TEXT NOT NULL,
  authors_json TEXT NOT NULL DEFAULT '[]',
  published    TEXT DEFAULT '',
  updated      TEXT DEFAULT '',
  primary_category TEXT DEFAULT '',
  categories_json  TEXT NOT NULL DEFAULT '[]',
  comment TEXT DEFAULT '',
  journal_ref TEXT DEFAULT '',
  doi TEXT DEFAULT '',
  abs_url      TEXT NOT NULL DEFAULT '',
  pdf_url      TEXT DEFAULT '',
  fetched_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_embeddings (
  paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  model    TEXT NOT NULL,
  dim      INTEGER NOT NULL,
  vector   BLOB NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (paper_id, model)
);

CREATE TABLE IF NOT EXISTS landscapes (
  id INTEGER PRIMARY KEY,
  topic_id INTEGER NOT NULL REFERENCES topics(id),
  title TEXT NOT NULL,
  summary TEXT DEFAULT '',
  status TEXT NOT NULL DEFAULT 'running',
  params_json TEXT NOT NULL DEFAULT '{}',
  generation INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS landscape_papers (
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  paper_id TEXT NOT NULL REFERENCES papers(paper_id),
  rank INTEGER NOT NULL,
  relevance_score REAL NOT NULL,
  cross_encoder_logit REAL,
  rerank_source TEXT NOT NULL DEFAULT 'cross-encoder',
  rationale TEXT DEFAULT '',
  is_seed INTEGER NOT NULL DEFAULT 0,
  cluster_id INTEGER,
  x REAL,
  y REAL,
  added_at TEXT NOT NULL,
  PRIMARY KEY (landscape_id, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_landscape_papers_rank ON landscape_papers(landscape_id, rank);

CREATE TABLE IF NOT EXISTS clusters (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  local_label INTEGER NOT NULL,
  label TEXT DEFAULT '',
  description TEXT DEFAULT '',
  paper_count INTEGER NOT NULL DEFAULT 0,
  x REAL DEFAULT 0,
  y REAL DEFAULT 0,
  color TEXT DEFAULT '#94a3b8'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clusters_unique ON clusters(landscape_id, local_label);

CREATE TABLE IF NOT EXISTS edges (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  src_paper_id TEXT NOT NULL,
  dst_paper_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 0.5,
  rationale TEXT DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
  ON edges(landscape_id, src_paper_id, dst_paper_id, kind);

CREATE TABLE IF NOT EXISTS tensions (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  statement TEXT NOT NULL,
  paper_a_id TEXT DEFAULT '',
  paper_b_id TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS open_problems (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  statement TEXT NOT NULL,
  why_open TEXT DEFAULT '',
  supporting_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS reading_path (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  paper_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  why TEXT DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reading_path_unique ON reading_path(landscape_id, position);

CREATE TABLE IF NOT EXISTS paper_extractions (
  paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  prompt_version TEXT NOT NULL,
  model TEXT NOT NULL DEFAULT '',
  problem TEXT,
  method TEXT,
  results TEXT,
  contribution TEXT,
  limitations TEXT,
  datasets_json TEXT NOT NULL DEFAULT '[]',
  metrics_json TEXT NOT NULL DEFAULT '[]',
  novelty TEXT DEFAULT 'unclear',
  evidence_json TEXT NOT NULL DEFAULT '{}',
  confidence REAL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (paper_id, prompt_version)
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  landscape_id INTEGER REFERENCES landscapes(id) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  message TEXT DEFAULT '',
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_landscape ON runs(landscape_id, created_at);
"""


class StoreError(RuntimeError):
    """Raised when a write cannot be completed."""


# --------------------------------------------------------------------------- #
# Connection handling
# --------------------------------------------------------------------------- #


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a WAL-mode connection with row access by name.

    ``check_same_thread=False`` plus one connection per unit of work is what
    makes it safe to call store functions from the worker threads that
    ``asyncio.to_thread`` uses for the blocking arXiv client and model calls.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


#: Columns added after the first release. ``CREATE TABLE IF NOT EXISTS`` cannot
#: add a column to an existing table, so additive changes are applied explicitly
#: here; a user's database outlives any single schema change.
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("landscape_papers", "cross_encoder_logit", "REAL"),
)


def _apply_additive_columns(conn: sqlite3.Connection) -> None:
    for table, column, column_type in _ADDITIVE_COLUMNS:
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if not existing:  # table itself is missing; CREATE handled it
            continue
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def init_db(db_path: Path | str) -> None:
    """Apply the schema. Idempotent: safe to call on every startup."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _apply_additive_columns(conn)
        conn.commit()
    finally:
        conn.close()


def _resolve(settings: Settings | None) -> Settings:
    return settings or load_settings(require_llm=False)


@contextmanager
def session(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    """Unit-of-work boundary: commit on success, roll back on failure."""
    resolved = _resolve(settings)
    resolved.ensure_dirs()
    conn = connect(resolved.db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_load(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #


def upsert_topic(conn: sqlite3.Connection, query_text: str) -> int:
    """Return the topic id for ``query_text``, creating it if needed.

    Matching is case- and whitespace-insensitive via the unique index, so
    "RAG" and "rag " resolve to the same topic and can accumulate papers.
    """
    normalized = " ".join(query_text.split())
    row = conn.execute(
        "SELECT id FROM topics WHERE lower(trim(query_text)) = lower(trim(?))",
        (normalized,),
    ).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO topics (query_text, created_at) VALUES (?, ?)",
        (normalized, utcnow()),
    )
    return int(cur.lastrowid)


def fetch_topic(conn: sqlite3.Connection, topic_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Papers
# --------------------------------------------------------------------------- #


def upsert_papers(conn: sqlite3.Connection, papers: Sequence[Paper]) -> int:
    """Insert or refresh paper metadata. Returns the number of new rows.

    Metadata is refreshed on conflict because abstracts are occasionally revised
    upstream; the extraction cache keys on ``prompt_version`` rather than on the
    abstract text, so a revised abstract is picked up on the next extraction run
    only when the prompt version is bumped.
    """
    if not papers:
        return 0
    now = utcnow()
    existing = {
        row["paper_id"]
        for row in conn.execute(
            f"SELECT paper_id FROM papers WHERE paper_id IN ({_placeholders(len(papers))})",
            [p.paper_id for p in papers],
        )
    }
    conn.executemany(
        """
        INSERT INTO papers (
            paper_id, version, title, abstract, authors_json, published, updated,
            primary_category, categories_json, comment, journal_ref, doi,
            abs_url, pdf_url, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            version = excluded.version,
            title = excluded.title,
            abstract = excluded.abstract,
            authors_json = excluded.authors_json,
            published = excluded.published,
            updated = excluded.updated,
            primary_category = excluded.primary_category,
            categories_json = excluded.categories_json,
            comment = excluded.comment,
            journal_ref = excluded.journal_ref,
            doi = excluded.doi,
            abs_url = excluded.abs_url,
            pdf_url = excluded.pdf_url,
            fetched_at = excluded.fetched_at
        """,
        [
            (
                p.paper_id,
                p.version,
                p.title,
                p.abstract,
                _json_dump(p.authors),
                p.published,
                p.updated,
                p.primary_category,
                _json_dump(p.categories),
                p.comment,
                p.journal_ref,
                p.doi,
                p.abs_link,
                p.pdf_url,
                now,
            )
            for p in papers
        ],
    )
    return len([p for p in papers if p.paper_id not in existing])


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        paper_id=row["paper_id"],
        version=row["version"] or "",
        title=row["title"],
        abstract=row["abstract"],
        authors=_json_load(row["authors_json"], []),
        published=row["published"] or "",
        updated=row["updated"] or "",
        primary_category=row["primary_category"] or "",
        categories=_json_load(row["categories_json"], []),
        comment=row["comment"] or "",
        journal_ref=row["journal_ref"] or "",
        doi=row["doi"] or "",
        abs_url=row["abs_url"] or "",
        pdf_url=row["pdf_url"] or "",
    )


def fetch_paper(conn: sqlite3.Connection, paper_id: str) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return _row_to_paper(row) if row else None


def fetch_papers(conn: sqlite3.Connection, paper_ids: Sequence[str]) -> dict[str, Paper]:
    if not paper_ids:
        return {}
    rows = conn.execute(
        f"SELECT * FROM papers WHERE paper_id IN ({_placeholders(len(paper_ids))})",
        list(paper_ids),
    )
    return {row["paper_id"]: _row_to_paper(row) for row in rows}


def count_papers(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"])


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def upsert_embeddings(
    conn: sqlite3.Connection,
    model: str,
    vectors: dict[str, bytes],
) -> None:
    """Store raw float32 payloads. ``vectors`` maps paper_id -> BLOB bytes."""
    if not vectors:
        return
    now = utcnow()
    conn.executemany(
        """
        INSERT INTO paper_embeddings (paper_id, model, dim, vector, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, model) DO UPDATE SET
            dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at
        """,
        [(pid, model, len(blob) // 4, blob, now) for pid, blob in vectors.items()],
    )


def fetch_embeddings(
    conn: sqlite3.Connection, model: str, paper_ids: Sequence[str] | None = None
) -> dict[str, bytes]:
    if paper_ids is None:
        rows = conn.execute(
            "SELECT paper_id, vector FROM paper_embeddings WHERE model = ? ORDER BY paper_id",
            (model,),
        )
    else:
        if not paper_ids:
            return {}
        rows = conn.execute(
            f"SELECT paper_id, vector FROM paper_embeddings WHERE model = ? "
            f"AND paper_id IN ({_placeholders(len(paper_ids))})",
            (model, *paper_ids),
        )
    return {row["paper_id"]: bytes(row["vector"]) for row in rows}


# --------------------------------------------------------------------------- #
# Landscapes
# --------------------------------------------------------------------------- #


def insert_landscape(
    conn: sqlite3.Connection, *, topic_id: int, title: str, params: dict[str, Any] | None = None
) -> int:
    now = utcnow()
    cur = conn.execute(
        """
        INSERT INTO landscapes (topic_id, title, summary, status, params_json,
                                generation, created_at, updated_at)
        VALUES (?, ?, '', 'running', ?, 1, ?, ?)
        """,
        (topic_id, title, _json_dump(params or {}), now, now),
    )
    return int(cur.lastrowid)


def fetch_landscape(conn: sqlite3.Connection, landscape_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT l.*, t.query_text AS topic
        FROM landscapes l JOIN topics t ON t.id = l.topic_id
        WHERE l.id = ?
        """,
        (landscape_id,),
    ).fetchone()
    return dict(row) if row else None


def find_landscape_for_topic(conn: sqlite3.Connection, topic_id: int) -> dict[str, Any] | None:
    """Most recent landscape for a topic, used by ``expand`` and by re-runs."""
    row = conn.execute(
        """
        SELECT l.*, t.query_text AS topic
        FROM landscapes l JOIN topics t ON t.id = l.topic_id
        WHERE l.topic_id = ?
        ORDER BY l.generation DESC, l.id DESC
        LIMIT 1
        """,
        (topic_id,),
    ).fetchone()
    return dict(row) if row else None


def list_landscapes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT l.*, t.query_text AS topic,
               (SELECT COUNT(*) FROM landscape_papers lp WHERE lp.landscape_id = l.id) AS paper_count,
               (SELECT COUNT(*) FROM clusters c
                 WHERE c.landscape_id = l.id AND c.local_label != ?) AS cluster_count
        FROM landscapes l JOIN topics t ON t.id = l.topic_id
        ORDER BY l.updated_at DESC, l.id DESC
        """,
        (UNCLUSTERED_LABEL,),
    )
    return [dict(row) for row in rows]


def update_landscape(
    conn: sqlite3.Connection,
    landscape_id: int,
    *,
    title: str | None = None,
    summary: str | None = None,
    status: str | None = None,
) -> None:
    sets: list[str] = ["updated_at = ?"]
    params: list[Any] = [utcnow()]
    if title is not None:
        sets.append("title = ?")
        params.append(title)
    if summary is not None:
        sets.append("summary = ?")
        params.append(summary)
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    params.append(landscape_id)
    conn.execute(f"UPDATE landscapes SET {', '.join(sets)} WHERE id = ?", params)


def bump_generation(conn: sqlite3.Connection, landscape_id: int) -> int:
    conn.execute(
        "UPDATE landscapes SET generation = generation + 1, updated_at = ? WHERE id = ?",
        (utcnow(), landscape_id),
    )
    row = conn.execute("SELECT generation FROM landscapes WHERE id = ?", (landscape_id,)).fetchone()
    return int(row["generation"]) if row else 1


def delete_landscape(conn: sqlite3.Connection, landscape_id: int) -> None:
    conn.execute("DELETE FROM landscapes WHERE id = ?", (landscape_id,))


# --------------------------------------------------------------------------- #
# Landscape membership and layout
# --------------------------------------------------------------------------- #


def link_paper(
    conn: sqlite3.Connection,
    landscape_id: int,
    *,
    paper_id: str,
    rank: int,
    relevance_score: float,
    rerank_source: str,
    rationale: str = "",
    is_seed: bool = False,
    cross_encoder_logit: float | None = None,
) -> None:
    """Attach a paper to a landscape, preserving any existing x/y position."""
    conn.execute(
        """
        INSERT INTO landscape_papers (landscape_id, paper_id, rank, relevance_score,
                                      cross_encoder_logit, rerank_source, rationale,
                                      is_seed, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(landscape_id, paper_id) DO UPDATE SET
            rank = excluded.rank,
            relevance_score = excluded.relevance_score,
            cross_encoder_logit = excluded.cross_encoder_logit,
            rerank_source = excluded.rerank_source,
            rationale = excluded.rationale,
            is_seed = excluded.is_seed
        """,
        (
            landscape_id,
            paper_id,
            rank,
            relevance_score,
            cross_encoder_logit,
            rerank_source,
            rationale,
            1 if is_seed else 0,
            utcnow(),
        ),
    )


def fetch_landscape_papers(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT lp.rank, lp.relevance_score, lp.cross_encoder_logit, lp.rerank_source,
               lp.rationale, lp.is_seed, lp.cluster_id, lp.x, lp.y, lp.added_at,
               p.paper_id, p.version, p.title, p.abstract, p.authors_json,
               p.published, p.primary_category, p.categories_json, p.abs_url, p.pdf_url
        FROM landscape_papers lp JOIN papers p ON p.paper_id = lp.paper_id
        WHERE lp.landscape_id = ?
        ORDER BY lp.rank ASC, p.paper_id ASC
        """,
        (landscape_id,),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["authors"] = _json_load(item.pop("authors_json"), [])
        item["categories"] = _json_load(item.pop("categories_json"), [])
        item["is_seed"] = bool(item["is_seed"])
        out.append(item)
    return out


def landscape_paper_ids(conn: sqlite3.Connection, landscape_id: int) -> set[str]:
    rows = conn.execute(
        "SELECT paper_id FROM landscape_papers WHERE landscape_id = ?", (landscape_id,)
    )
    return {row["paper_id"] for row in rows}


def update_layout(
    conn: sqlite3.Connection,
    landscape_id: int,
    layout: dict[str, tuple[float, float, int]],
) -> None:
    """Persist coordinates and cluster assignment: paper_id -> (x, y, cluster)."""
    if not layout:
        return
    conn.executemany(
        "UPDATE landscape_papers SET x = ?, y = ?, cluster_id = ? "
        "WHERE landscape_id = ? AND paper_id = ?",
        [(x, y, cluster, landscape_id, pid) for pid, (x, y, cluster) in layout.items()],
    )


# --------------------------------------------------------------------------- #
# Clusters
# --------------------------------------------------------------------------- #


def replace_clusters(
    conn: sqlite3.Connection, landscape_id: int, clusters: Sequence[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM clusters WHERE landscape_id = ?", (landscape_id,))
    if not clusters:
        return
    conn.executemany(
        """
        INSERT INTO clusters (landscape_id, local_label, label, description,
                              paper_count, x, y, color)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                landscape_id,
                int(c["local_label"]),
                c.get("label", ""),
                c.get("description", ""),
                int(c.get("paper_count", 0)),
                float(c.get("x", 0.0)),
                float(c.get("y", 0.0)),
                c.get("color", "#94a3b8"),
            )
            for c in clusters
        ],
    )


def fetch_clusters(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM clusters WHERE landscape_id = ? ORDER BY local_label ASC",
        (landscape_id,),
    )
    return [dict(row) for row in rows]


def cluster_id_by_label(conn: sqlite3.Connection, landscape_id: int) -> dict[int, int]:
    """Map HDBSCAN's local label (-1, 0, 1, ...) to the clusters table row id."""
    rows = conn.execute(
        "SELECT id, local_label FROM clusters WHERE landscape_id = ?", (landscape_id,)
    )
    return {int(row["local_label"]): int(row["id"]) for row in rows}


# --------------------------------------------------------------------------- #
# Edges, tensions, open problems, reading path
# --------------------------------------------------------------------------- #


def replace_edges(conn: sqlite3.Connection, landscape_id: int, edges: Sequence[EdgeClaim]) -> None:
    conn.execute("DELETE FROM edges WHERE landscape_id = ?", (landscape_id,))
    if not edges:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO edges (landscape_id, src_paper_id, dst_paper_id, kind, weight, rationale)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (landscape_id, e.src_paper_id, e.dst_paper_id, e.kind, float(e.weight), e.rationale)
            for e in edges
        ],
    )


def fetch_edges(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT src_paper_id, dst_paper_id, kind, weight, rationale FROM edges "
        "WHERE landscape_id = ? ORDER BY id ASC",
        (landscape_id,),
    )
    return [dict(row) for row in rows]


def replace_tensions(conn: sqlite3.Connection, landscape_id: int, tensions: Sequence[Tension]) -> None:
    conn.execute("DELETE FROM tensions WHERE landscape_id = ?", (landscape_id,))
    if not tensions:
        return
    conn.executemany(
        "INSERT INTO tensions (landscape_id, statement, paper_a_id, paper_b_id) VALUES (?, ?, ?, ?)",
        [(landscape_id, t.statement, t.paper_a_id, t.paper_b_id) for t in tensions],
    )


def fetch_tensions(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT statement, paper_a_id, paper_b_id FROM tensions WHERE landscape_id = ? ORDER BY id ASC",
        (landscape_id,),
    )
    return [dict(row) for row in rows]


def replace_open_problems(
    conn: sqlite3.Connection, landscape_id: int, problems: Sequence[OpenProblem]
) -> None:
    conn.execute("DELETE FROM open_problems WHERE landscape_id = ?", (landscape_id,))
    if not problems:
        return
    conn.executemany(
        "INSERT INTO open_problems (landscape_id, statement, why_open, supporting_json) "
        "VALUES (?, ?, ?, ?)",
        [(landscape_id, p.statement, p.why_open, _json_dump(p.supporting_paper_ids)) for p in problems],
    )


def fetch_open_problems(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT statement, why_open, supporting_json FROM open_problems "
        "WHERE landscape_id = ? ORDER BY id ASC",
        (landscape_id,),
    )
    out = []
    for row in rows:
        item = dict(row)
        item["supporting_paper_ids"] = _json_load(item.pop("supporting_json"), [])
        out.append(item)
    return out


def replace_reading_path(
    conn: sqlite3.Connection, landscape_id: int, steps: Sequence[ReadingStep]
) -> None:
    conn.execute("DELETE FROM reading_path WHERE landscape_id = ?", (landscape_id,))
    if not steps:
        return
    conn.executemany(
        "INSERT INTO reading_path (landscape_id, paper_id, position, why) VALUES (?, ?, ?, ?)",
        [(landscape_id, s.paper_id, int(s.position), s.why) for s in steps],
    )


def fetch_reading_path(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT rp.paper_id, rp.position, rp.why, p.title
        FROM reading_path rp LEFT JOIN papers p ON p.paper_id = rp.paper_id
        WHERE rp.landscape_id = ?
        ORDER BY rp.position ASC
        """,
        (landscape_id,),
    )
    return [dict(row) for row in rows]


def replace_cluster_labels(
    conn: sqlite3.Connection, landscape_id: int, labels: Sequence[ClusterLabel]
) -> None:
    """Apply LLM labels onto rows already written by the layout stage."""
    for label in labels:
        conn.execute(
            "UPDATE clusters SET label = ?, description = ? WHERE landscape_id = ? AND local_label = ?",
            (label.label, label.description, landscape_id, int(label.local_label)),
        )


# --------------------------------------------------------------------------- #
# Extractions
# --------------------------------------------------------------------------- #


def upsert_extraction(
    conn: sqlite3.Connection,
    paper_id: str,
    prompt_version: str,
    extraction: PaperExtraction,
    *,
    model: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO paper_extractions (
            paper_id, prompt_version, model, problem, method, results, contribution,
            limitations, datasets_json, metrics_json, novelty, evidence_json,
            confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, prompt_version) DO UPDATE SET
            model = excluded.model,
            problem = excluded.problem,
            method = excluded.method,
            results = excluded.results,
            contribution = excluded.contribution,
            limitations = excluded.limitations,
            datasets_json = excluded.datasets_json,
            metrics_json = excluded.metrics_json,
            novelty = excluded.novelty,
            evidence_json = excluded.evidence_json,
            confidence = excluded.confidence,
            created_at = excluded.created_at
        """,
        (
            paper_id,
            prompt_version,
            model,
            extraction.problem,
            extraction.method,
            extraction.results,
            extraction.contribution,
            extraction.limitations,
            _json_dump(extraction.datasets),
            _json_dump(extraction.metrics),
            extraction.novelty,
            _json_dump(extraction.evidence),
            extraction.confidence,
            utcnow(),
        ),
    )


def _row_to_extraction(row: sqlite3.Row) -> PaperExtraction:
    return PaperExtraction(
        problem=row["problem"],
        method=row["method"],
        results=row["results"],
        contribution=row["contribution"],
        limitations=row["limitations"],
        datasets=_json_load(row["datasets_json"], []),
        metrics=_json_load(row["metrics_json"], []),
        novelty=row["novelty"] or "unclear",
        evidence=_json_load(row["evidence_json"], {}),
        confidence=row["confidence"],
    )


def fetch_extraction(
    conn: sqlite3.Connection, paper_id: str, prompt_version: str
) -> PaperExtraction | None:
    row = conn.execute(
        "SELECT * FROM paper_extractions WHERE paper_id = ? AND prompt_version = ?",
        (paper_id, prompt_version),
    ).fetchone()
    return _row_to_extraction(row) if row else None


def fetch_extractions(
    conn: sqlite3.Connection, paper_ids: Sequence[str], prompt_version: str
) -> dict[str, PaperExtraction]:
    if not paper_ids:
        return {}
    rows = conn.execute(
        f"SELECT * FROM paper_extractions WHERE prompt_version = ? "
        f"AND paper_id IN ({_placeholders(len(paper_ids))})",
        (prompt_version, *paper_ids),
    )
    return {row["paper_id"]: _row_to_extraction(row) for row in rows}


# --------------------------------------------------------------------------- #
# Runs (SSE replay)
# --------------------------------------------------------------------------- #


def insert_run(conn: sqlite3.Connection, event_payload: dict[str, Any]) -> None:
    """Record a stage event so a client that reconnects can replay progress."""
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (id, landscape_id, stage, status, message, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_payload["id"],
            event_payload.get("landscape_id"),
            event_payload["stage"],
            event_payload["status"],
            event_payload.get("message", ""),
            _json_dump(event_payload.get("payload", {})),
            event_payload.get("created_at") or utcnow(),
        ),
    )


def fetch_runs(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs WHERE landscape_id = ? ORDER BY created_at ASC, id ASC",
        (landscape_id,),
    )
    return [dict(row) for row in rows]


def default_settings() -> Settings:
    """Convenience for scripts that only need paths, not credentials."""
    settings = Settings.from_env()
    settings.ensure_dirs()
    return settings


__all__ = [
    "PROJECT_ROOT",
    "SCHEMA",
    "StoreError",
    "bump_generation",
    "cluster_id_by_label",
    "connect",
    "count_papers",
    "default_settings",
    "delete_landscape",
    "fetch_clusters",
    "fetch_edges",
    "fetch_embeddings",
    "fetch_extraction",
    "fetch_extractions",
    "fetch_landscape",
    "fetch_landscape_papers",
    "fetch_open_problems",
    "fetch_paper",
    "fetch_papers",
    "fetch_reading_path",
    "fetch_runs",
    "fetch_tensions",
    "fetch_topic",
    "find_landscape_for_topic",
    "init_db",
    "insert_landscape",
    "insert_run",
    "landscape_paper_ids",
    "link_paper",
    "list_landscapes",
    "replace_cluster_labels",
    "replace_clusters",
    "replace_edges",
    "replace_open_problems",
    "replace_reading_path",
    "replace_tensions",
    "session",
    "update_landscape",
    "update_layout",
    "upsert_embeddings",
    "upsert_extraction",
    "upsert_papers",
    "upsert_topic",
]
