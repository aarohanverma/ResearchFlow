"""Async SQLAlchemy engine and session factory.
One engine, one factory — shared across the process lifetime."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,       # detect stale connections before use
    pool_size=10,
    max_overflow=20,
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
