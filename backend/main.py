"""ResearchFlow FastAPI application entry point.

Layer order: lifespan → middleware → routers → exception handlers.
No business logic here — just wiring.
"""

import logging
import os
import time
import uuid as _uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as v1_router
from app.core.config import settings
from app.scheduler.jobs import start_scheduler, stop_scheduler

_LOG_LEVEL = (
    logging.DEBUG
    if settings.debug
    else getattr(logging, settings.log_level, logging.INFO)
)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


_GUEST_EMAIL = "test@researchflow.ai"
_GUEST_PASSWORD = "ResearchFlow2024!"


async def _ensure_seed_user() -> None:
    """Idempotently create the guest/test user and its SourceMappings on startup."""
    from sqlalchemy import select as _select
    from app.db.session import async_session_factory
    from app.core.security import hash_password
    from app.models.user import User, UserProviderSettings, UserInterestProfile, ExpertiseLevel, Orientation
    from app.models.graph import NamespaceSubscription, SourceMapping

    _DEFAULT_NS = [
        ("cs.AI",  "arxiv_rss", "cs.AI"),
        ("cs.ML",  "arxiv_rss", "cs.LG"),
        ("cs.NLP", "arxiv_rss", "cs.CL"),
    ]

    async with async_session_factory() as db:
        row = await db.execute(_select(User).where(User.email == _GUEST_EMAIL))
        user = row.scalar_one_or_none()

        if not user:
            user = User(
                email=_GUEST_EMAIL,
                hashed_password=hash_password(_GUEST_PASSWORD),
                display_name="Guest Researcher",
                expertise_level=ExpertiseLevel.practitioner,
                orientation=Orientation.both,
                onboarding_complete=True,
            )
            db.add(user)
            await db.flush()
            db.add(UserProviderSettings(user_id=user.id))
            db.add(UserInterestProfile(user_id=user.id))
            log.info("seed user created: %s", _GUEST_EMAIL)

        for ns_key, source_name, arxiv_cat in _DEFAULT_NS:
            sub = await db.execute(
                _select(NamespaceSubscription).where(
                    NamespaceSubscription.user_id == user.id,
                    NamespaceSubscription.namespace_key == ns_key,
                )
            )
            if not sub.scalar_one_or_none():
                db.add(NamespaceSubscription(user_id=user.id, namespace_key=ns_key))

            mapping = await db.execute(
                _select(SourceMapping).where(
                    SourceMapping.namespace_key == ns_key,
                    SourceMapping.source_name == source_name,
                )
            )
            if not mapping.scalar_one_or_none():
                db.add(SourceMapping(
                    namespace_key=ns_key,
                    source_name=source_name,
                    external_category_key=arxiv_cat,
                ))

        await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Startup: create DB tables (dev), start scheduler. Shutdown: stop scheduler."""
    log.info("ResearchFlow starting — environment=%s debug=%s", settings.environment, settings.debug)

    if settings.environment == "local":
        from app.db.session import create_all_tables
        await create_all_tables()

    # Create search indexes idempotently — safe to run every startup.
    # GIN for full-text search, HNSW for vector similarity, composite for filtering.
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_papers_fts
                ON papers USING GIN (
                    to_tsvector('english',
                        COALESCE(title, '') || ' ' ||
                        COALESCE(tldr, '') || ' ' ||
                        COALESCE(abstract, '') || ' ' ||
                        COALESCE(array_to_string(key_concepts, ' '), '') || ' ' ||
                        COALESCE(array_to_string(methods_used, ' '), '')
                    )
                )
            """))
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_papers_external_id
                ON papers (external_id)
            """))
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_chunks_dim_provider
                ON paper_chunks (embedding_dim, embedding_provider)
            """))
            # HNSW index for fast approximate nearest-neighbour search.
            # Requires pgvector >= 0.5. Skipped silently if unavailable.
            try:
                await conn.execute(_text("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                    ON paper_chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """))
            except Exception:
                pass  # pgvector HNSW not available — IVFFlat/exact scan fallback
        log.info("search indexes ensured")
    except Exception as exc:
        log.warning("search index creation skipped: %s", exc)

    # Schema migrations — ADD COLUMN IF NOT EXISTS is idempotent
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text2
        async with engine.begin() as conn:
            await conn.execute(_text2("""
                ALTER TABLE idea_capsules
                ADD COLUMN IF NOT EXISTS source_mode VARCHAR(20) NOT NULL DEFAULT 'manual'
            """))
            await conn.execute(_text2("""
                ALTER TABLE idea_capsules
                ADD COLUMN IF NOT EXISTS source_query TEXT
            """))
        log.info("idea_capsules schema migration complete")
    except Exception as exc:
        log.warning("idea_capsules migration skipped: %s", exc)

    # Ensure the guest/test user always exists in local dev — idempotent
    if settings.environment == "local":
        try:
            await _ensure_seed_user()
        except Exception as exc:
            log.warning("seed user creation skipped: %s", exc)

    start_scheduler()
    log.info("scheduler started")

    yield

    stop_scheduler()
    log.info("ResearchFlow shutting down")


app = FastAPI(
    title="ResearchFlow API",
    description="AI-native research intelligence platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing + request-ID middleware ─────────────────────────────────────
@app.middleware("http")
async def request_instrumentation(request: Request, call_next):
    req_id = str(_uuid.uuid4())[:8]
    request.state.request_id = req_id
    t0 = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

    log_fn = log.debug if settings.debug else log.info
    log_fn(
        "%s %s → %d  %sms  req_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        req_id,
    )
    if settings.debug and request.query_params:
        log.debug("  query=%s", dict(request.query_params))

    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(v1_router)

# ── Static files (blob storage in local dev) ──────────────────────────────────
blob_dir = settings.blob_local_dir
os.makedirs(blob_dir, exist_ok=True)
app.mount("/blobs", StaticFiles(directory=blob_dir, check_dir=False), name="blobs")


# ── Global exception handler — never leak internals ───────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(
        "unhandled exception path=%s req_id=%s err=%s",
        request.url.path,
        getattr(request.state, "request_id", "?"),
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Something went wrong. We're on it.",
            "action": "Try again or contact support.",
        },
    )


# ── Health + debug endpoints ───────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment}


@app.get("/debug/status", include_in_schema=settings.debug)
async def debug_status():
    """Runtime config snapshot — only accessible when DEBUG=true."""
    if not settings.debug:
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    def _mask(val: str) -> str:
        return val[:4] + "…" if len(val) > 8 else ("set" if val else "")

    return {
        "environment": settings.environment,
        "log_level": settings.log_level,
        "debug": settings.debug,
        "database_url": settings.database_url.split("@")[-1],  # hide credentials
        "cache_backend": settings.cache_backend,
        "blob_backend": settings.blob_backend,
        "ingestion_mode": settings.ingestion_mode,
        "default_llm_provider": settings.default_llm_provider,
        "default_embedding_provider": settings.default_embedding_provider,
        "default_embedding_model": settings.default_embedding_model,
        "openai_api_key": _mask(settings.openai_api_key),
        "google_api_key": _mask(settings.google_api_key),
        "anthropic_api_key": _mask(settings.anthropic_api_key),
        "langsmith_tracing": bool(settings.langsmith_api_key),
        "breakthrough_threshold": settings.breakthrough_threshold,
        "ingestion_cron": settings.ingestion_cron,
    }
