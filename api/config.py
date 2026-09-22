"""Central configuration for the research landscape agent.

All tunables live here so pipeline behavior can be adjusted without touching
pipeline logic. Mirrors the pattern used in RAG-Pipeline-With-Hybrid-Search.

Paths in the environment are resolved relative to the project root (the parent
of this file's directory) so that the CLI, the API server, and the tests all
agree on where ``data/`` lives regardless of the working directory they are
launched from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when the environment is missing something required to run."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _str(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    """Resolve a path setting, relative entries anchored at the project root."""
    raw = _str(name, default)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


VALID_PROVIDERS = {"nim", "openrouter"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the environment."""

    # --- LLM ---
    llm_provider: str = "nim"
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Deliberately not a guess: this slug is the one already in use in the
    # RAG-Pipeline project. Confirm against GET /v1/models before a real run.
    llm_model: str = "meta/llama-3.1-70b-instruct"
    llm_concurrency: int = 4
    llm_max_repairs: int = 2
    llm_timeout_seconds: int = 120

    # --- Retrieval ---
    arxiv_delay_seconds: float = 3.0
    arxiv_num_retries: int = 5
    arxiv_page_size: int = 100
    retrieval_max_results: int = 200
    retrieval_cache_ttl_hours: int = 24
    retrieval_cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/cache/arxiv")
    arxiv_offline: bool = False
    arxiv_force_429: bool = False

    # --- Reranking ---
    rerank_seed_count: int = 40
    rerank_final_count: int = 60
    rerank_blend_ce: float = 0.6
    rerank_blend_llm: float = 0.4
    rerank_judge_batch_size: int = 10
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: str = "cpu"
    cross_encoder_max_length: int = 512
    cross_encoder_batch_size: int = 16

    # --- Embeddings / layout ---
    embed_model: str = "BAAI/bge-base-en-v1.5"
    umap_random_state: int = 42
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 0  # 0 => auto
    # HDBSCAN defaults ``min_samples`` to ``min_cluster_size``, which is far more
    # conservative than a reading map can afford: on a real 60-paper landscape
    # that default labelled 24 papers (40%) as noise -- one grey blob. Measured
    # across both the real layout and the synthetic fixtures, 3 is the lowest
    # value that keeps well-separated structure intact (at 2, three clean blobs
    # split into four clusters and a lone outlier becomes its own cluster).
    hdbscan_min_samples: int = 3

    # --- Storage ---
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data/landscapes.db")
    prompt_version: str = "extract_v1"

    # --- Server ---
    api_port: int = 8000
    allowed_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the process environment (and ``.env``)."""
        provider = _str("LLM_PROVIDER", "nim").lower()
        if provider not in VALID_PROVIDERS:
            # An unknown provider is a typo, not a request for the default.
            raise ConfigError(
                f"LLM_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got {provider!r}"
            )
        return cls(
            llm_provider=provider,
            nvidia_api_key=_str("NVIDIA_API_KEY", ""),
            nvidia_base_url=_str("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"),
            openrouter_api_key=_str("OPENROUTER_API_KEY", ""),
            openrouter_base_url=_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            llm_model=_str("LLM_MODEL", "meta/llama-3.1-70b-instruct"),
            llm_concurrency=_int("LLM_CONCURRENCY", 4),
            llm_max_repairs=_int("LLM_MAX_REPAIRS", 2),
            llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 120),
            arxiv_delay_seconds=_float("ARXIV_DELAY_SECONDS", 3.0),
            arxiv_num_retries=_int("ARXIV_NUM_RETRIES", 5),
            arxiv_page_size=_int("ARXIV_PAGE_SIZE", 100),
            retrieval_max_results=_int("RETRIEVAL_MAX_RESULTS", 200),
            retrieval_cache_ttl_hours=_int("RETRIEVAL_CACHE_TTL_HOURS", 24),
            retrieval_cache_dir=_path("RETRIEVAL_CACHE_DIR", "./data/cache/arxiv"),
            arxiv_offline=_bool("ARXIV_OFFLINE", False),
            arxiv_force_429=_bool("ARXIV_FORCE_429", False),
            rerank_seed_count=_int("RERANK_SEED_COUNT", 40),
            rerank_final_count=_int("RERANK_FINAL_COUNT", 60),
            rerank_blend_ce=_float("RERANK_BLEND_CE", 0.6),
            rerank_blend_llm=_float("RERANK_BLEND_LLM", 0.4),
            rerank_judge_batch_size=_int("RERANK_JUDGE_BATCH_SIZE", 10),
            cross_encoder_model=_str("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            cross_encoder_device=_str("CROSS_ENCODER_DEVICE", "cpu"),
            cross_encoder_max_length=_int("CROSS_ENCODER_MAX_LENGTH", 512),
            cross_encoder_batch_size=_int("CROSS_ENCODER_BATCH_SIZE", 16),
            embed_model=_str("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
            umap_random_state=_int("UMAP_RANDOM_STATE", 42),
            umap_n_neighbors=_int("UMAP_N_NEIGHBORS", 15),
            umap_min_dist=_float("UMAP_MIN_DIST", 0.1),
            hdbscan_min_cluster_size=_int("HDBSCAN_MIN_CLUSTER_SIZE", 0),
            hdbscan_min_samples=max(1, _int("HDBSCAN_MIN_SAMPLES", 3)),
            db_path=_path("DB_PATH", "./data/landscapes.db"),
            prompt_version=_str("PROMPT_VERSION", "extract_v1"),
            api_port=_int("API_PORT", 8000),
            allowed_origins=_str("ALLOWED_ORIGINS", "http://localhost:3000"),
            log_level=_str("LOG_LEVEL", "INFO").upper(),
        )

    # --- Derived ---

    @property
    def llm_base_url(self) -> str:
        return self.nvidia_base_url if self.llm_provider == "nim" else self.openrouter_base_url

    @property
    def llm_api_key(self) -> str:
        return self.nvidia_api_key if self.llm_provider == "nim" else self.openrouter_api_key

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def model_profile_dir(self) -> Path:
        """Local HuggingFace cache root, so model downloads land inside data/."""
        return self.db_path.parent / "models"

    @property
    def rerank_final_count_capped(self) -> int:
        """Never ask for more final papers than we retrieved."""
        return min(self.rerank_final_count, self.retrieval_max_results)

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_profile_dir.mkdir(parents=True, exist_ok=True)
        self.apply_model_cache_env()

    def apply_model_cache_env(self) -> None:
        """Point HuggingFace at ``data/models`` before any model is imported.

        Without this, the cross-encoder and embedding weights land in the user's
        global ``~/.cache/huggingface``, which makes the project's footprint
        invisible and awkward to clean up. ``setdefault`` so an explicit
        environment setting still wins.
        """
        os.environ.setdefault("HF_HOME", str(self.model_profile_dir))
        os.environ.setdefault(
            "SENTENCE_TRANSFORMERS_HOME", str(self.model_profile_dir / "sentence_transformers")
        )

    def validate(self, *, require_llm: bool = True) -> None:
        """Fail early and name the missing variable.

        Tests run without an LLM key, so ``require_llm=False`` is supported.
        """
        problems: list[str] = []
        if require_llm and not self.llm_api_key:
            var = "NVIDIA_API_KEY" if self.llm_provider == "nim" else "OPENROUTER_API_KEY"
            problems.append(
                f"{var} is not set (LLM_PROVIDER={self.llm_provider}). "
                "Copy .env.example to .env and fill it in."
            )
        if require_llm and not self.llm_model:
            problems.append("LLM_MODEL is empty. Confirm a model slug from GET /v1/models.")
        if self.arxiv_delay_seconds < 3.0:
            problems.append(
                "ARXIV_DELAY_SECONDS must be >= 3.0 per the arXiv Terms of Use "
                "(no more than one request every three seconds)."
            )
        if self.rerank_final_count < 1:
            problems.append("RERANK_FINAL_COUNT must be at least 1.")
        blend = self.rerank_blend_ce + self.rerank_blend_llm
        if abs(blend - 1.0) > 1e-6:
            problems.append(
                f"RERANK_BLEND_CE + RERANK_BLEND_LLM must sum to 1.0, got {blend}."
            )
        if not 0.0 < self.rerank_blend_ce < 1.0:
            problems.append("RERANK_BLEND_CE must be strictly between 0 and 1.")
        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))

    def describe(self) -> dict[str, object]:
        """Redacted view of the configuration, safe to log."""
        out: dict[str, object] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if "key" in f.name:
                value = "set" if value else "unset"
            out[f.name] = value
        return out


def load_settings(*, require_llm: bool = False) -> Settings:
    """Load, validate, and materialize directories in one call."""
    settings = Settings.from_env()
    settings.validate(require_llm=require_llm)
    settings.ensure_dirs()
    return settings
