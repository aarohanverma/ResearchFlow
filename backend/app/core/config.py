"""Central settings — loaded once at startup via Pydantic BaseSettings.
All values come from environment variables or .env.local.  No hardcoded secrets."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at backend/app/core/config.py → project root is 4 levels up
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env.local"


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables and ``.env.local``.

    All secrets and deployment-specific values are injected via environment
    variables; no hardcoded secrets appear here. ``get_settings()`` caches a
    singleton instance via ``@lru_cache`` so this class is only instantiated once.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    environment: Literal["local", "azure"] = "local"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://researchflow:researchflow@localhost:5432/researchflow"
    )

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_backend: Literal["local", "redis"] = "local"
    cache_dir: str = str(Path.home() / ".cache" / "researchflow")
    redis_url: str = "redis://localhost:6379/0"

    # ── Blob Storage ──────────────────────────────────────────────────────────
    blob_backend: Literal["local", "azure"] = "local"
    blob_local_dir: str = str(Path.home() / ".cache" / "researchflow" / "blobs")
    azure_storage_connection_string: str = ""

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080  # 7 days

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    default_cheap_model: str = "gpt-4o-mini"
    default_quality_model: str = "gpt-5.4-mini"
    default_reasoning_model: str = "gpt-5.4"
    default_llm_provider: Literal["openai", "anthropic", "google"] = "openai"

    # ── Embeddings ────────────────────────────────────────────────────────────
    default_embedding_provider: Literal["gemini", "openai", "voyage"] = "gemini"
    default_embedding_model: str = "gemini-embedding-2-preview"
    default_embedding_dim: int = 768
    voyage_api_key: str = ""

    # ── Image Generation ──────────────────────────────────────────────────────
    image_gen_provider: str = "openai"

    # ── PDF Parsing ───────────────────────────────────────────────────────────
    pdf_parser: Literal["marker", "gemini_vision"] = "marker"
    marker_api_key: str = ""

    # ── Ingestion ─────────────────────────────────────────────────────────────
    ingestion_mode: Literal["rss", "mcp"] = "rss"
    arxiv_mcp_transport: Literal["stdio", "sse"] = "stdio"
    arxiv_mcp_command: str = "uv run arxiv-mcp-server"
    arxiv_mcp_url: str = "http://localhost:8765/sse"

    # ── Email ─────────────────────────────────────────────────────────────────
    resend_api_key: str = ""
    email_from: str = "noreply@researchflow.ai"
    email_from_name: str = "ResearchFlow"

    # ── Observability ─────────────────────────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "researchflow"
    langchain_tracing_v2: bool = True

    # ── Web Search (LLM tool) ─────────────────────────────────────────────────
    web_search_provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    tavily_api_key: str = ""

    # ── Thresholds ────────────────────────────────────────────────────────────
    breakthrough_threshold: float = Field(default=0.88, ge=0.0, le=1.0)

    # ── Scheduler crons (standard cron syntax) ────────────────────────────────
    ingestion_cron: str = "59 23 * * *"
    clustering_cron: str = "0 2 * * 0"
    cross_namespace_cron: str = "0 3 * * 0"


@lru_cache
def get_settings() -> Settings:
    """Return the cached ``Settings`` singleton.

    The instance is constructed once on first call and reused on all
    subsequent calls thanks to ``@lru_cache``.

    Returns:
        The application-wide ``Settings`` instance.
    """
    return Settings()


settings = get_settings()
