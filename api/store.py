"""SQLite persistence layer.

Design notes:

* **stdlib ``sqlite3``, no ORM.** Every table's DDL is explicit and reviewable
  here, so there is nothing to invent and no dependency-version risk.
* **WAL mode + a connection per operation.** SQLite connections are not
  shareable across threads, and this pipeline mixes the event-loop thread with
  ``asyncio.to_thread`` workers (the arXiv client, cross-encoder, and UMAP are
  all synchronous). Opening a short-lived connection per call is cheap and
  sidesteps the whole class of "SQLite objects created in a thread can only be
  used in that same thread" errors.
* **Rows come back as dicts.** Callers build Pydantic models from them.
* **JSON columns are TEXT.** SQLite has no native JSON type; ``_dumps``/``_loads``
  keep the encoding in one place.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from config import PROJECT_ROOT, Settings
from models import Paper, PaperExtraction, utcnow

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS topics (
  id INTEGER PRIMARY KEY,
  query_text TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_query
  ON topics(lower(trim(query_text)));

CREATE TABLE IF NOT EXISTS papers (
  paper_id     TEXT PRIMARY KEY,
  version      TEXT NOT NULL DEFAULT '',
  title        TEXT NOT NULL,
  abstract     TEXT NOT NULL,
  authors_json TEXT NOT NULL DEFAULT '[]',
  published    TEXT NOT NULL DEFAULT '',
  updated      TEXT NOT NULL DEFAULT '',
  primary_category TEXT NOT NULL DEFAULT '',
  categories_json  TEXT NOT NULL DEFAULT '[]',
  comment      TEXT NOT NULL DEFAULT '',
  journal_ref  TEXT NOT NULL DEFAULT '',
  doi          TEXT NOT NULL DEFAULT '',
  abs_url      TEXT NOT NULL DEFAULT '',
  pdf_url      TEXT NOT NULL DEFAULT '',
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
  summary TEXT NOT NULL DEFAULT '',
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
  rerank_source TEXT NOT NULL DEFAULT 'cross-encoder',
  rationale TEXT NOT NULL DEFAULT '',
  is_seed INTEGER NOT NULL DEFAULT 0,
  cluster_id INTEGER,
  x REAL,
  y REAL,
  added_at TEXT NOT NULL,
  PRIMARY KEY (landscape_id, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_landscape_papers_rank
  ON landscape_papers(landscape_id, rank);

CREATE TABLE IF NOT EXISTS clusters (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  local_label INTEGER NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  paper_count INTEGER NOT NULL DEFAULT 0,
  x REAL,
  y REAL,
  color TEXT NOT NULL DEFAULT '#94a3b8'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clusters_unique
  ON clusters(landscape_id, local_label);

CREATE TABLE IF NOT EXISTS edges (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  src_paper_id TEXT NOT NULL,
  dst_paper_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 0.5,
  rationale TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
  ON edges(landscape_id, src_paper_id, dst_paper_id, kind);

CREATE TABLE IF NOT EXISTS tensions (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  statement TEXT NOT NULL,
  paper_a_id TEXT NOT NULL DEFAULT '',
  paper_b_id TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS open_problems (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  statement TEXT NOT NULL,
  why_open TEXT NOT NULL DEFAULT '',
  supporting_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS reading_path (
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  paper_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  why TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (landscape_id, paper_id)
);

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
  novelty TEXT NOT NULL DEFAULT 'unclear',
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
  message TEXT NOT NULL DEFAULT '',
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_landscape
  ON runs(landscape_id, created_at);
"""


# --------------------------------------------------------------------------- #
# Connection plumbing
# --------------------------------------------------------------------------- #


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def connect(db_path: str | Any | None = None) -> sqlite3.Connection:
    """Open a connection with the pragmas this schema depends on."""
    target = db_path if db_path is not None else Settings.from_env().db_path
    if target != ":memory:":
        from pathlib import Path

        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(db_path: str | Any | None = None) -> None:
    """Apply the schema. Idempotent: safe to call on every startup."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def session(db_path: str | Any | None = None) -> Iterator[sqlite3.Connection]:
    """Transactional connection: commits on success, rolls back on error."""
    conn = connect(db_path)
    try:
        # executescript() implicitly commits, so make sure the schema exists
        # before any transaction is opened on a fresh database.
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def rows_to_dicts(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #


def upsert_topic(conn: sqlite3.Connection, query_text: str) -> int:
    """Return the topic id for this text, creating it if new.

    Matching is case- and whitespace-insensitive, mirroring the unique index.
    """
    cleaned = " ".join(query_text.split())
    row = conn.execute(
        "SELECT id FROM topics WHERE lower(trim(query_text)) = lower(trim(?))",
        (cleaned,),
    ).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO topics (query_text, created_at) VALUES (?, ?)",
        (cleaned, utcnow()),
    )
    return int(cur.lastrowid)


def fetch_topic_text(conn: sqlite3.Connection, topic_id: int) -> str:
    row = conn.execute("SELECT query_text FROM topics WHERE id = ?", (topic_id,)).fetchone()
    return str(row["query_text"]) if row else ""


# --------------------------------------------------------------------------- #
# Papers
# --------------------------------------------------------------------------- #


def upsert_papers(conn: sqlite3.Connection, papers: Sequence[Paper]) -> int:
    """Insert or refresh paper metadata. Returns the number of rows written."""
    if not papers:
        return 0
    now = utcnow()
    payload = [
        (
            p.paper_id,
            p.version,
            p.title,
            p.abstract,
            _dumps(p.authors),
            p.published,
            p.updated,
            p.primary_category,
            _dumps(p.categories),
            p.comment or "",
            p.journal_ref or "",
            p.doi or "",
            p.abs_link,
            p.pdf_url or "",
            now,
        )
        for p in papers
    ]
    conn.executemany(
        """
        INSERT INTO papers (
            paper_id, version, title, abstract, authors_json, published, updated,
            primary_category, categories_json, comment, journal_ref, doi,
            abs_url, pdf_url, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            version          = excluded.version,
            title            = excluded.title,
            abstract         = excluded.abstract,
            authors_json     = excluded.authors_json,
            published        = excluded.published,
            updated          = excluded.updated,
            primary_category = excluded.primary_category,
            categories_json  = excluded.categories_json,
            comment          = excluded.comment,
            journal_ref      = excluded.journal_ref,
            doi              = excluded.doi,
            abs_url          = excluded.abs_url,
            pdf_url          = excluded.pdf_url,
            fetched_at       = excluded.fetched_at
        """,
        payload,
    )
    return len(payload)


def _paper_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "version": row["version"],
        "title": row["title"],
        "abstract": row["abstract"],
        "authors": _loads(row["authors_json"], []),
        "published": row["published"],
        "updated": row["updated"],
        "primary_category": row["primary_category"],
        "categories": _loads(row["categories_json"], []),
        "comment": row["comment"],
        "journal_ref": row["journal_ref"],
        "doi": row["doi"],
        "abs_url": row["abs_url"],
        "pdf_url": row["pdf_url"],
    }


def fetch_paper(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return _paper_from_row(row) if row else None


def fetch_papers(conn: sqlite3.Connection, paper_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not paper_ids:
        return []
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"SELECT * FROM papers WHERE paper_id IN ({placeholders})",  # noqa: S608 - placeholder count is fixed
        list(paper_ids),
    ).fetchall()
    return [_paper_from_row(r) for r in rows]


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def upsert_embedding(
    conn: sqlite3.Connection,
    paper_id: str,
    model: str,
    vector: bytes,
    dim: int,
) -> None:
    conn.execute(
        """
        INSERT INTO paper_embeddings (paper_id, model, dim, vector, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, model) DO UPDATE SET
            dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at
        """,
        (paper_id, model, dim, vector, utcnow()),
    )


def upsert_embeddings(
    conn: sqlite3.Connection,
    model: str,
    items: Sequence[tuple[str, bytes, int]],
) -> None:
    if not items:
        return
    now = utcnow()
    conn.executemany(
        """
        INSERT INTO paper_embeddings (paper_id, model, dim, vector, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, model) DO UPDATE SET
            dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at
        """,
        [(pid, model, dim, vec, now) for pid, vec, dim in items],
    )


def fetch_embeddings(
    conn: sqlite3.Connection, model: str, paper_ids: Sequence[str] | None = None
) -> dict[str, bytes]:
    if paper_ids:
        placeholders = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"SELECT paper_id, vector FROM paper_embeddings WHERE model = ? AND paper_id IN ({placeholders})",  # noqa: S608
            [model, *paper_ids],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT paper_id, vector FROM paper_embeddings WHERE model = ?", (model,)
        ).fetchall()
    return {row["paper_id"]: row["vector"] for row in rows}


def count_embeddings(conn: sqlite3.Connection, model: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM paper_embeddings WHERE model = ?", (model,)
    ).fetchone()
    return int(row["n"]) if row else 0


# --------------------------------------------------------------------------- #
# Landscapes
# --------------------------------------------------------------------------- #


def insert_landscape(
    conn: sqlite3.Connection,
    topic_id: int,
    title: str,
    params: dict[str, Any] | None = None,
) -> int:
    now = utcnow()
    cur = conn.execute(
        """
        INSERT INTO landscapes (topic_id, title, summary, status, params_json,
                                generation, created_at, updated_at)
        VALUES (?, ?, '', 'running', ?, 1, ?, ?)
        """,
        (topic_id, title, _dumps(params or {}), now, now),
    )
    return int(cur.lastrowid)


def fetch_landscape_row(conn: sqlite3.Connection, landscape_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT l.*, t.query_text AS topic
        FROM landscapes l JOIN topics t ON t.id = l.topic_id
        WHERE l.id = ?
        """,
        (landscape_id,),
    ).fetchone()
    if not row:
        return None
    data = dict(row)
    data["params"] = _loads(data.pop("params_json", "{}"), {})
    return data


def list_landscapes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT l.id, l.title, l.summary, l.status, l.generation,
               l.created_at, l.updated_at, t.query_text AS topic,
               (SELECT COUNT(*) FROM landscape_papers lp WHERE lp.landscape_id = l.id) AS paper_count,
               (SELECT COUNT(*) FROM clusters c
                 WHERE c.landscape_id = l.id AND c.local_label >= 0) AS cluster_count
        FROM landscapes l JOIN topics t ON t.id = l.topic_id
        ORDER BY l.updated_at DESC, l.id DESC
        """
    ).fetchall()
    return rows_to_dicts(rows)


def set_landscape_status(
    conn: sqlite3.Connection,
    landscape_id: int,
    status: str,
    *,
    error: str | None = None,
) -> None:
    if error is not None:
        conn.execute(
            "UPDATE landscapes SET status = ?, summary = ?, updated_at = ? WHERE id = ?",
            (status, error, utcnow(), landscape_id),
        )
        return
    conn.execute(
        "UPDATE landscapes SET status = ?, updated_at = ? WHERE id = ?",
        (status, utcnow(), landscape_id),
    )


def update_landscape_meta(
    conn: sqlite3.Connection, landscape_id: int, *, title: str, summary: str
) -> None:
    conn.execute(
        "UPDATE landscapes SET title = ?, summary = ?, updated_at = ? WHERE id = ?",
        (title, summary, utcnow(), landscape_id),
    )


def bump_generation(conn: sqlite3.Connection, landscape_id: int) -> int:
    """Increment and return the new generation.

    Positions are always recomputed from scratch (never ``umap.transform()``ed),
    so the generation is what the UI uses to detect that the layout moved.
    """
    conn.execute(
        "UPDATE landscapes SET generation = generation + 1, updated_at = ? WHERE id = ?",
        (utcnow(), landscape_id),
    )
    row = conn.execute(
        "SELECT generation FROM landscapes WHERE id = ?", (landscape_id,)
    ).fetchone()
    return int(row["generation"]) if row else 1


def delete_landscape(conn: sqlite3.Connection, landscape_id: int) -> bool:
    cur = conn.execute("DELETE FROM landscapes WHERE id = ?", (landscape_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Landscape <-> paper links
# --------------------------------------------------------------------------- #


def link_papers(conn: sqlite3.Connection, landscape_id: int, ranked: Sequence[Any]) -> int:
    """Attach ranked papers to a landscape.

    ``INSERT OR IGNORE`` is deliberate: growing a landscape must never reset the
    ``added_at`` or the layout of a paper that is already on the map.
    """
    if not ranked:
        return 0
    now = utcnow()
    conn.executemany(
        """
        INSERT OR IGNORE INTO landscape_papers (
            landscape_id, paper_id, rank, relevance_score, rerank_source,
            rationale, is_seed, cluster_id, x, y, added_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
        """,
        [
            (
                landscape_id,
                item.paper.paper_id,
                item.rank,
                item.relevance_score,
                item.rerank_source,
                item.rationale,
                1 if item.rank <= 10 else 0,
                now,
            )
            for item in ranked
        ],
    )
    return len(ranked)


def fetch_landscape_papers(
    conn: sqlite3.Connection, landscape_id: int
) -> list[dict[str, Any]]:
    """Papers on the map, joined with their paper metadata, ordered by rank."""
    rows = conn.execute(
        """
        SELECT lp.rank, lp.relevance_score, lp.rerank_source, lp.rationale,
               lp.is_seed, lp.cluster_id, lp.x, lp.y, p.*
        FROM landscape_papers lp
        JOIN papers p ON p.paper_id = lp.paper_id
        WHERE lp.landscape_id = ?
        ORDER BY lp.rank ASC, lp.paper_id ASC
        """,
        (landscape_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        data = _paper_from_row(row)
        data.update(
            rank=int(row["rank"]),
            relevance_score=float(row["relevance_score"]),
            rerank_source=row["rerank_source"],
            rationale=row["rationale"],
            is_seed=bool(row["is_seed"]),
            cluster_id=row["cluster_id"],
            x=row["x"],
            y=row["y"],
        )
        out.append(data)
    return out


def existing_paper_ids(conn: sqlite3.Connection, landscape_id: int) -> set[str]:
    rows = conn.execute(
        "SELECT paper_id FROM landscape_papers WHERE landscape_id = ?", (landscape_id,)
    ).fetchall()
    return {str(r["paper_id"]) for r in rows}


def update_paper_layout(
    conn: sqlite3.Connection,
    landscape_id: int,
    entries: Sequence[tuple[str, int | None, float, float]],
) -> None:
    """Write cluster assignment and 2D coordinates for each paper."""
    if not entries:
        return
    conn.executemany(
        "UPDATE landscape_papers SET cluster_id = ?, x = ?, y = ? "
        "WHERE landscape_id = ? AND paper_id = ?",
        [(cluster_id, x, y, landscape_id, paper_id) for paper_id, cluster_id, x, y in entries],
    )


# --------------------------------------------------------------------------- #
# Clusters, edges, tensions, open problems, reading path
# --------------------------------------------------------------------------- #


def replace_clusters(conn: sqlite3.Connection, landscape_id: int, clusters: Sequence[dict[str, Any]]) -> None:
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
                c.get("x"),
                c.get("y"),
                c.get("color", "#94a3b8"),
            )
            for c in clusters
        ],
    )


def fetch_clusters(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM clusters WHERE landscape_id = ? ORDER BY local_label ASC",
        (landscape_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def replace_edges(conn: sqlite3.Connection, landscape_id: int, edges: Sequence[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM edges WHERE landscape_id = ?", (landscape_id,))
    if not edges:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO edges (landscape_id, src_paper_id, dst_paper_id, kind, weight, rationale)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                landscape_id,
                e["src_paper_id"],
                e["dst_paper_id"],
                e["kind"],
                float(e.get("weight", 0.5)),
                e.get("rationale", ""),
            )
            for e in edges
        ],
    )


def fetch_edges(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT src_paper_id, dst_paper_id, kind, weight, rationale FROM edges WHERE landscape_id = ?",
        (landscape_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def replace_tensions(conn: sqlite3.Connection, landscape_id: int, tensions: Sequence[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM tensions WHERE landscape_id = ?", (landscape_id,))
    if not tensions:
        return
    conn.executemany(
        "INSERT INTO tensions (landscape_id, statement, paper_a_id, paper_b_id) VALUES (?, ?, ?, ?)",
        [
            (landscape_id, t["statement"], t.get("paper_a_id", ""), t.get("paper_b_id", ""))
            for t in tensions
        ],
    )


def fetch_tensions(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT statement, paper_a_id, paper_b_id FROM tensions WHERE landscape_id = ? ORDER BY id ASC",
        (landscape_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def replace_open_problems(
    conn: sqlite3.Connection, landscape_id: int, problems: Sequence[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM open_problems WHERE landscape_id = ?", (landscape_id,))
    if not problems:
        return
    conn.executemany(
        """
        INSERT INTO open_problems (landscape_id, statement, why_open, supporting_json)
        VALUES (?, ?, ?, ?)
        """,
        [
            (
                landscape_id,
                p["statement"],
                p.get("why_open", ""),
                _dumps(p.get("supporting_paper_ids", [])),
            )
            for p in problems
        ],
    )


def fetch_open_problems(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT statement, why_open, supporting_json FROM open_problems WHERE landscape_id = ? ORDER BY id ASC",
        (landscape_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        data["supporting_paper_ids"] = _loads(data.pop("supporting_json", "[]"), [])
        out.append(data)
    return out


def replace_reading_path(conn: sqlite3.Connection, landscape_id: int, steps: Sequence[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM reading_path WHERE landscape_id = ?", (landscape_id,))
    if not steps:
        return
    conn.executemany(
        "INSERT INTO reading_path (landscape_id, paper_id, position, why) VALUES (?, ?, ?, ?)",
        [
            (landscape_id, s["paper_id"], int(s["position"]), s.get("why", ""))
            for s in steps
        ],
    )


def fetch_reading_path(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT rp.paper_id, rp.position, rp.why, p.title
        FROM reading_path rp
        LEFT JOIN papers p ON p.paper_id = rp.paper_id
        WHERE rp.landscape_id = ?
        ORDER BY rp.position ASC
        """,
        (landscape_id,),
    ).fetchall()
    return rows_to_dicts(rows)


# --------------------------------------------------------------------------- #
# Extractions
# --------------------------------------------------------------------------- #


def upsert_extraction(
    conn: sqlite3.Connection,
    paper_id: str,
    prompt_version: str,
    model: str,
    extraction: PaperExtraction,
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
            _dumps(extraction.datasets),
            _dumps(extraction.metrics),
            extraction.novelty,
            _dumps(extraction.evidence),
            extraction.confidence,
            utcnow(),
        ),
    )


def _extraction_from_row(row: sqlite3.Row) -> PaperExtraction:
    return PaperExtraction(
        problem=row["problem"],
        method=row["method"],
        results=row["results"],
        contribution=row["contribution"],
        limitations=row["limitations"],
        datasets=_loads(row["datasets_json"], []),
        metrics=_loads(row["metrics_json"], []),
        novelty=row["novelty"] or "unclear",
        evidence=_loads(row["evidence_json"], {}),
        confidence=row["confidence"],
    )


def fetch_extraction(
    conn: sqlite3.Connection, paper_id: str, prompt_version: str
) -> PaperExtraction | None:
    row = conn.execute(
        "SELECT * FROM paper_extractions WHERE paper_id = ? AND prompt_version = ?",
        (paper_id, prompt_version),
    ).fetchone()
    return _extraction_from_row(row) if row else None


def fetch_extractions(
    conn: sqlite3.Connection, prompt_version: str, paper_ids: Sequence[str]
) -> dict[str, PaperExtraction]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"SELECT * FROM paper_extractions WHERE prompt_version = ? AND paper_id IN ({placeholders})",  # noqa: S608
        [prompt_version, *paper_ids],
    ).fetchall()
    return {row["paper_id"]: _extraction_from_row(row) for row in rows}


def count_extractions(conn: sqlite3.Connection, prompt_version: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM paper_extractions WHERE prompt_version = ?",
        (prompt_version,),
    ).fetchone()
    return int(row["n"]) if row else 0


# --------------------------------------------------------------------------- #
# Run log
# --------------------------------------------------------------------------- #


def insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    landscape_id: int | None,
    stage: str,
    status: str,
    message: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (id, landscape_id, stage, status, message, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = excluded.status, message = excluded.message,
            payload_json = excluded.payload_json
        """,
        (run_id, landscape_id, stage, status, message, _dumps(payload or {}), utcnow()),
    )


def fetch_runs(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs WHERE landscape_id = ? ORDER BY created_at ASC",
        (landscape_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        data["payload"] = _loads(data.pop("payload_json", "{}"), {})
        out.append(data)
    return out


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def default_db_path() -> Any:
    return PROJECT_ROOT / "data/landscapes.db"


def settings_db_path(settings: Settings) -> Any:
    return settings.db_path
