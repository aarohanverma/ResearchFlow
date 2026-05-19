"""Async SQLAlchemy engine and session factory.
One engine, one factory — shared across the process lifetime."""

import json
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings


def _json_default(obj):
    # JSONB columns regularly carry UUIDs, timestamps, and Decimals from
    # repositories — stdlib json.dumps cannot serialize those by default,
    # which previously crashed the assistant finalize step.
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _json_serializer(value) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False)


engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,       # detect stale connections before use
    pool_size=10,
    max_overflow=20,
    # Recycle at 20 min — well under Azure PostgreSQL's 30-min server-side
    # idle timeout.  Setting recycle == idle_timeout means we sometimes
    # hand back a connection that the server already closed, causing the
    # next caller to get an error even with pool_pre_ping=True (pre-ping
    # only fires when the connection is checked OUT, not during idle wait).
    pool_recycle=1200,
    json_serializer=_json_serializer,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,   # avoid lazy-load errors after commit in async context
)


async def create_all_tables() -> None:
    """Dev-only table creation used during startup when Alembic is not yet run."""
    from app.db.base import Base  # noqa: F401
    import app.models  # noqa: F401 — registers all models on Base.metadata

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

        # Idempotent schema evolution (no Alembic)
        await conn.execute(text("ALTER TABLE papers ADD COLUMN IF NOT EXISTS tldr TEXT"))

        # bookmark_folders / bookmark_folder_members created by create_all above
        # but if the bookmarks table predates us we need to ensure folder_id
        # column does NOT exist (we use the junction table instead). Drop it
        # if it was added by a previous migration attempt.
        await conn.execute(text(
            "ALTER TABLE bookmarks DROP COLUMN IF EXISTS folder_id"
        ))
