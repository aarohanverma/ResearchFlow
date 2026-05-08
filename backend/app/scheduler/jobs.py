"""APScheduler jobs — nightly ingestion, weekly clustering, weekly cross-namespace links."""

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.config import settings

log = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None


async def _run_ingestion_all() -> None:
    """Run ingestion for every SourceMapping namespace in the DB."""
    from app.db.session import async_session_factory
    from app.repositories.graph import GraphRepository
    from app.workflows.ingestion import run_all_ingestion

    async with async_session_factory() as db:
        repo = GraphRepository(db)
        mappings = await repo.get_all_source_mappings()
        namespaces = list({m.namespace_key for m in mappings})

    if not namespaces:
        log.info("scheduler.ingestion: no namespaces configured — skipping")
        return

    log.info("scheduler.ingestion: running for %d namespaces", len(namespaces))
    await run_all_ingestion(namespaces)


async def _rebuild_bookmark_index() -> None:
    """Re-embed all bookmarked papers that are missing an abstract chunk."""
    from app.adapters.embedding import get_embedding_adapter
    from app.db.session import async_session_factory
    from app.models.paper import PaperChunk
    from app.repositories.paper import PaperRepository
    from sqlalchemy import text as sa_text

    log.info("scheduler.bookmark_index_rebuild: starting")
    try:
        embed = get_embedding_adapter()
        async with async_session_factory() as db:
            rows = await db.execute(sa_text("SELECT DISTINCT user_id FROM bookmarks"))
            user_ids = [r[0] for r in rows.fetchall()]

        rebuilt = 0
        for uid in user_ids:
            async with async_session_factory() as db:
                repo = PaperRepository(db)
                bookmarks = await repo.get_bookmarks(uid)
                for bm in bookmarks:
                    chunks = await repo.get_chunks(bm.paper_id)
                    if any(c.section_type == "abstract" for c in chunks):
                        continue
                    paper = await repo.get_by_id(bm.paper_id)
                    if not paper:
                        continue
                    try:
                        vectors = await embed.embed_texts([paper.abstract], task_type="RETRIEVAL_DOCUMENT")
                        db.add(PaperChunk(
                            paper_id=paper.id,
                            chunk_index=0,
                            section_type="abstract",
                            content=paper.abstract,
                            embedding=vectors[0],
                            embedding_dim=embed.dimensions,
                            embedding_provider=embed.provider_id,
                        ))
                        rebuilt += 1
                    except Exception as exc:
                        log.warning("bookmark_index_rebuild: embed failed paper=%s err=%s", paper.id, exc)
                await db.commit()

        log.info("scheduler.bookmark_index_rebuild: done rebuilt=%d", rebuilt)
    except Exception as exc:
        log.error("scheduler.bookmark_index_rebuild: failed err=%s", exc)




async def _run_clustering() -> None:
    """Placeholder for the weekly subtopic-discovery clustering job (HDBSCAN, post-MVP)."""
    log.info("scheduler.clustering: starting")
    # Clustering workflow would go here — subtopic discovery via HDBSCAN
    # Stub: actual implementation deferred to ClusteringWorkflow (post-MVP)
    pass


async def _run_cross_namespace_links() -> None:
    """Placeholder for the weekly cross-namespace concept-bridge edge job (post-MVP)."""
    log.info("scheduler.cross_namespace: starting")
    # Weekly cross-namespace similarity edges — deferred to full implementation
    pass


def start_scheduler() -> None:
    """Initialise and start the APScheduler instance with all configured cron jobs.

    Creates an ``AsyncIOScheduler`` and registers four recurring jobs:

    - **ingestion_nightly**: runs ``_run_ingestion_all`` on the cron schedule
      defined by ``settings.ingestion_cron``.
    - **clustering_weekly**: runs ``_run_clustering`` on the cron schedule
      defined by ``settings.clustering_cron``.
    - **cross_namespace_weekly**: runs ``_run_cross_namespace_links`` on the
      cron schedule defined by ``settings.cross_namespace_cron``.
    - **bookmark_index_rebuild_weekly**: runs ``_rebuild_bookmark_index``
      every Sunday at 03:00 UTC.

    This function is idempotent — if the scheduler is already running it
    returns immediately without creating a second instance.
    """
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = AsyncIOScheduler()

    # Parse cron strings from settings (format: "minute hour day month weekday")
    def _parse_cron(cron_str: str) -> dict:
        """Convert a 5-field cron string into an APScheduler keyword-argument dict."""
        parts = cron_str.split()
        keys = ["minute", "hour", "day", "month", "day_of_week"]
        return dict(zip(keys, parts))

    ingestion_cron = _parse_cron(settings.ingestion_cron)
    _scheduler.add_job(
        _run_ingestion_all,
        "cron",
        id="ingestion_nightly",
        **ingestion_cron,
        misfire_grace_time=3600,
    )

    clustering_cron = _parse_cron(settings.clustering_cron)
    _scheduler.add_job(
        _run_clustering,
        "cron",
        id="clustering_weekly",
        **clustering_cron,
    )

    xns_cron = _parse_cron(settings.cross_namespace_cron)
    _scheduler.add_job(
        _run_cross_namespace_links,
        "cron",
        id="cross_namespace_weekly",
        **xns_cron,
    )

    # Weekly nightly bookmark index rebuild — catches any papers that missed embedding
    _scheduler.add_job(
        _rebuild_bookmark_index,
        "cron",
        id="bookmark_index_rebuild_weekly",
        day_of_week="sun",
        hour=3,
        minute=0,
        misfire_grace_time=3600,
    )

    _scheduler.start()
    log.info("scheduler started: ingestion=%s clustering=%s", settings.ingestion_cron, settings.clustering_cron)


def stop_scheduler() -> None:
    """Shut down the APScheduler instance if it is currently running.

    Calls ``shutdown(wait=False)`` so the application can exit immediately
    without waiting for any in-progress jobs to finish, then clears the
    module-level reference. Safe to call when the scheduler is not running —
    does nothing in that case.
    """
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
