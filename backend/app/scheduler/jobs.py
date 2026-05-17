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
    """Re-embed all bookmarked papers that are missing an abstract chunk.

    Uses batch queries to find papers lacking abstract embeddings — a single
    LEFT JOIN query per user replaces the previous per-paper ``get_chunks``
    loop that issued one query per bookmarked paper.
    """
    from app.adapters.embedding import get_embedding_adapter
    from app.db.session import async_session_factory
    from app.models.paper import Bookmark, Paper, PaperChunk
    from app.repositories.paper import PaperRepository
    from sqlalchemy import distinct, select

    log.info("scheduler.bookmark_index_rebuild: starting")
    try:
        embed = get_embedding_adapter()
        async with async_session_factory() as db:
            rows = await db.execute(select(distinct(Bookmark.user_id)))
            user_ids = [r[0] for r in rows.fetchall()]

        rebuilt = 0
        for uid in user_ids:
            try:
                async with async_session_factory() as db:
                    repo = PaperRepository(db)
                    bookmarks = await repo.get_bookmarks(uid)
                    if not bookmarks:
                        continue

                    bm_paper_ids = [bm.paper_id for bm in bookmarks]

                    # Single query: find which bookmarked papers already have an abstract chunk
                    abstract_chunk_q = await db.execute(
                        select(PaperChunk.paper_id).where(
                            PaperChunk.paper_id.in_(bm_paper_ids),
                            PaperChunk.section_type == "abstract",
                        )
                    )
                    already_embedded: set = {row[0] for row in abstract_chunk_q.fetchall()}

                    # Batch-fetch only the papers that need embedding
                    missing_ids = [pid for pid in bm_paper_ids if pid not in already_embedded]
                    if not missing_ids:
                        continue

                    papers_q = await db.execute(
                        select(Paper).where(Paper.id.in_(missing_ids))
                    )
                    papers_to_embed = [p for p in papers_q.scalars() if p.abstract]

                    for paper in papers_to_embed:
                        try:
                            vectors = await embed.embed_texts(
                                [paper.abstract], task_type="RETRIEVAL_DOCUMENT"
                            )
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
                            log.warning(
                                "bookmark_index_rebuild: embed failed paper=%s err=%s",
                                paper.id, exc,
                            )

                    await db.commit()
            except Exception as exc:
                log.warning("bookmark_index_rebuild: failed for user=%s err=%s", uid, exc)

        log.info("scheduler.bookmark_index_rebuild: done rebuilt=%d", rebuilt)
    except Exception as exc:
        log.error("scheduler.bookmark_index_rebuild: failed err=%s", exc)




async def _cleanup_checkpoints() -> None:
    """Delete LangGraph checkpoint rows older than 30 days to prevent unbounded table growth."""
    log.info("scheduler.checkpoint_cleanup: starting")
    try:
        from app.db.checkpointer import get_checkpointer
        checkpointer = await get_checkpointer()
        removed = await checkpointer.cleanup_old_checkpoints(older_than_days=30)
        log.info("scheduler.checkpoint_cleanup: done removed=%d thread(s)", removed)
    except Exception as exc:
        log.error("scheduler.checkpoint_cleanup: failed err=%s", exc)


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

    # Build and configure the scheduler before assigning to the global so
    # that if start() raises (e.g. no running event loop on some platforms),
    # a subsequent call to start_scheduler() will retry rather than returning
    # early with a half-initialized, non-running scheduler reference.
    sched = AsyncIOScheduler()

    # Parse cron strings from settings (format: "minute hour day month weekday")
    def _parse_cron(cron_str: str) -> dict:
        """Convert a 5-field cron string into an APScheduler keyword-argument dict.

        Raises:
            ValueError: If ``cron_str`` does not contain exactly 5 fields.
        """
        parts = cron_str.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression {cron_str!r}: expected 5 fields "
                f"(minute hour day month weekday), got {len(parts)}"
            )
        keys = ["minute", "hour", "day", "month", "day_of_week"]
        return dict(zip(keys, parts))

    ingestion_cron = _parse_cron(settings.ingestion_cron)
    sched.add_job(
        _run_ingestion_all,
        "cron",
        id="ingestion_nightly",
        **ingestion_cron,
        misfire_grace_time=3600,
    )

    clustering_cron = _parse_cron(settings.clustering_cron)
    sched.add_job(
        _run_clustering,
        "cron",
        id="clustering_weekly",
        **clustering_cron,
        misfire_grace_time=3600,
    )

    xns_cron = _parse_cron(settings.cross_namespace_cron)
    sched.add_job(
        _run_cross_namespace_links,
        "cron",
        id="cross_namespace_weekly",
        **xns_cron,
        misfire_grace_time=3600,
    )

    # Weekly nightly bookmark index rebuild — catches any papers that missed embedding
    sched.add_job(
        _rebuild_bookmark_index,
        "cron",
        id="bookmark_index_rebuild_weekly",
        day_of_week="sun",
        hour=3,
        minute=0,
        misfire_grace_time=3600,
    )

    # Monthly LangGraph checkpoint cleanup — removes threads older than 30 days
    # to prevent unbounded growth of the three checkpoint tables.
    sched.add_job(
        _cleanup_checkpoints,
        "cron",
        id="checkpoint_cleanup_monthly",
        day=1,
        hour=4,
        minute=0,
        misfire_grace_time=3600,
    )

    sched.start()  # assign only after successful start
    _scheduler = sched
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
