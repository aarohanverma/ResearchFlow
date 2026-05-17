"""Central settings — loaded once at startup via Pydantic BaseSettings.
All values come from environment variables or .env.local.  No hardcoded secrets."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at backend/app/core/config.py → project root is 4 levels up
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# Accept .env.local (preferred for local dev) and .env (preferred for prod /
# deploys) so keys land in Settings whichever convention the operator uses.
# Later entries win, so .env.local overrides .env when both exist.
_ENV_FILES = [
    str(_PROJECT_ROOT / ".env"),
    str(_PROJECT_ROOT / ".env.local"),
]


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables and ``.env.local``.

    All secrets and deployment-specific values are injected via environment
    variables; no hardcoded secrets appear here. ``get_settings()`` caches a
    singleton instance via ``@lru_cache`` so this class is only instantiated once.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    environment: Literal["local", "azure"] = "local"
    debug: bool = False
    enable_dev_reset: bool = False
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
    # Container name for Azure Blob Storage. Operators must create this
    # container in the target storage account before flipping
    # BLOB_BACKEND=azure. Overridable per-deployment via AZURE_STORAGE_CONTAINER.
    azure_storage_container: str = "researchflow"

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
    default_embedding_provider: Literal["gemini", "openai", "voyage"] = "openai"
    default_embedding_model: str = "text-embedding-3-large"
    default_embedding_dim: int = 768  # OpenAI adapter requests 768-dim via Matryoshka truncation
    voyage_api_key: str = ""

    # ── Image Generation ──────────────────────────────────────────────────────
    image_gen_provider: str = "openai"

    # ── PDF Parsing ───────────────────────────────────────────────────────────
    # PDF parser default. Marker is the safest choice for low-resource hosts
    # (WSL, Docker on a laptop) — Docling pulls in PyTorch + EasyOCR which
    # can spike RAM enough to crash a small VM on first parse. Set
    # ``PDF_PARSER=docling`` explicitly when you want the richer structured
    # parser; the runtime fallback chain handles failures either way.
    pdf_parser: Literal["marker", "gemini_vision", "docling"] = "marker"
    marker_api_key: str = ""

    # ── Ingestion ─────────────────────────────────────────────────────────────
    ingestion_mode: Literal["rss", "mcp"] = "rss"
    arxiv_mcp_transport: Literal["stdio", "sse"] = "stdio"
    arxiv_mcp_command: str = "python -m arxiv_mcp_server --storage-path /data/papers"
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

    # ── Wolfram Alpha (RA computation tool) ───────────────────────────────────
    # MCP: docker run -i --rm -e WOLFRAM_ALPHA_APP_ID=<key> mcp/wolfram-alpha
    wolfram_mcp_command: str = ""
    wolfram_alpha_app_id: str = ""  # fallback direct API key

    # ── Thresholds ────────────────────────────────────────────────────────────
    breakthrough_threshold: float = Field(default=0.88, ge=0.0, le=1.0)

    # ── Scheduler crons (standard cron syntax) ────────────────────────────────
    # arXiv announces Mon–Thu 8 PM ET; RSS updates midnight ET; 1 AM ET = 05:00 UTC (EDT).
    # New content lands Tue–Fri mornings → run ingestion on days 2–5 only.
    # Weekly maintenance on Sunday (day 0), staggered 30 min.
    ingestion_cron: str = "0 5 * * 2-5"
    clustering_cron: str = "0 5 * * 0"
    cross_namespace_cron: str = "30 5 * * 0"

    # ── TTS (podcast generation) ──────────────────────────────────────────────
    tts_provider: str = "openai"
    tts_model: str = "tts-1-hd"

    # ── Slides (Marp) ─────────────────────────────────────────────────────────
    slides_provider: str = "marp"

    # ── Namespace-specific research tool keys ─────────────────────────────────
    fred_api_key: str = ""       # FRED macroeconomic data (econ/q-fin namespaces)
    ads_api_token: str = ""      # NASA ADS astronomy search (astro-ph namespace)
    nvd_api_key: str = ""        # NVD CVE database — improves rate limits; free without key
    github_token: str = ""       # GitHub search — improves rate limits; free personal token works


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
