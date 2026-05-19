"""WorkflowRepository — idempotency and token usage tracking."""

from datetime import date, datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workflow import TokenUsage, WorkflowRun, WorkflowStatus

# How long after ``started_at`` a ``running`` row is still considered to
# represent a live workflow, blocking a fresh trigger. After this window
# we assume the previous worker crashed / was killed and allow the next
# trigger to start a new run. 2 h matches the orchestrator's recovery
# window for assistant tasks (``recovery._RESUME_AGE_LIMIT``).
_IN_FLIGHT_GRACE = timedelta(hours=2)


class WorkflowRepository:
    """Data-access layer for ``WorkflowRun`` idempotency records and ``TokenUsage`` logs.

    Provides helpers to start, complete, and fail workflow runs, as well as
    recording per-call token consumption for the usage dashboard.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the repository with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for all queries.
        """
        self._db = db

    @staticmethod
    def _utc_midnight(d: date | None = None) -> datetime:
        """Convert a date to an explicit UTC-midnight datetime.

        Storing a bare Python ``date`` into a ``TIMESTAMP WITH TIME ZONE``
        column via asyncpg uses the *local* system timezone, which on IST
        machines shifts the stored value to the previous day in UTC.  Always
        binding an explicit UTC-aware datetime avoids that conversion.
        """
        if d is None:
            d = datetime.now(timezone.utc).date()
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    async def should_run(
        self, workflow_name: str, scope_key: str, run_date: date | None = None
    ) -> bool:
        """Return ``True`` if the workflow should run for the given day.

        Returns ``False`` when EITHER:
          * a ``completed`` run already exists today (the original idempotency
            guard), or
          * a ``running`` row exists that started within ``_IN_FLIGHT_GRACE``
            (concurrent-run guard — prevents double-billing when ingestion is
            triggered manually while a previous run is still mid-flight).

        Stale ``running`` rows older than the grace window are ignored so a
        crashed previous worker doesn't block legitimate retries forever.

        Args:
            workflow_name: Logical name of the workflow (e.g. ``"ingestion"``).
            scope_key: Scoping key, typically a namespace key (e.g. ``"cs.AI"``).
            run_date: The date to check against. Defaults to today in UTC when
                ``None``.

        Returns:
            ``True`` if neither a completed run for today nor a live in-flight
            run exists, meaning the workflow should execute.
        """
        run_dt = self._utc_midnight(run_date)
        in_flight_cutoff = datetime.now(timezone.utc) - _IN_FLIGHT_GRACE
        result = await self._db.execute(
            select(WorkflowRun).where(
                WorkflowRun.workflow_name == workflow_name,
                WorkflowRun.scope_key == scope_key,
                or_(
                    # Completed today — original idempotency guard.
                    (WorkflowRun.run_date == run_dt) &
                    (WorkflowRun.status == WorkflowStatus.completed),
                    # Currently in-flight within the grace window — concurrent
                    # trigger guard. Falls open once the row is older than the
                    # grace period so a crashed run doesn't deadlock retries.
                    (WorkflowRun.status == WorkflowStatus.running) &
                    (WorkflowRun.started_at >= in_flight_cutoff),
                ),
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is None

    async def start_run(self, workflow_name: str, scope_key: str) -> UUID:
        """Insert a new ``WorkflowRun`` row with status ``running`` and return its ID.

        Args:
            workflow_name: Logical name of the workflow (e.g. ``"ingestion"``).
            scope_key: Scoping key for the run, typically a namespace key
                (e.g. ``"cs.AI"``).

        Returns:
            The UUID of the newly created ``WorkflowRun`` row.
        """
        run = WorkflowRun(
            workflow_name=workflow_name,
            scope_key=scope_key,
            run_date=self._utc_midnight(),
            status=WorkflowStatus.running,
        )
        self._db.add(run)
        await self._db.flush()
        return run.id

    async def mark_completed(
        self, run_id: UUID, content_hash: str | None = None
    ) -> None:
        """Mark a workflow run as successfully completed.

        Args:
            run_id: UUID of the ``WorkflowRun`` to update.
            content_hash: Optional hash of the run's output content, used for
                change-detection on future runs. Defaults to ``None``.
        """
        await self._db.execute(
            update(WorkflowRun)
            .where(WorkflowRun.id == run_id)
            .values(
                status=WorkflowStatus.completed,
                content_hash=content_hash,
                completed_at=datetime.now(timezone.utc),
            )
        )

    async def mark_failed(self, run_id: UUID, error_metadata: dict) -> None:
        """Mark a workflow run as failed and store error details.

        Args:
            run_id: UUID of the ``WorkflowRun`` to update.
            error_metadata: Arbitrary dict of error information (e.g. exception
                messages keyed by node name) to persist alongside the run.
        """
        await self._db.execute(
            update(WorkflowRun)
            .where(WorkflowRun.id == run_id)
            .values(
                status=WorkflowStatus.failed,
                error_metadata=error_metadata,
                completed_at=datetime.now(timezone.utc),
            )
        )

    async def record_token_usage(self, usage: dict) -> None:
        """Persist a single ``TokenUsage`` row from a dict of column values.

        Args:
            usage: Dictionary mapping ``TokenUsage`` column names to values
                (e.g. ``provider``, ``model``, ``input_tokens``,
                ``output_tokens``, ``cost_usd``, ``latency_ms``).
        """
        self._db.add(TokenUsage(**usage))
        await self._db.flush()

    async def get_usage_summary(self, user_id: UUID) -> list[dict]:
        """Return the most recent token-usage records for a user.

        Fetches up to 200 rows ordered by ``created_at`` descending, returning
        each row as a plain dict with keys ``provider``, ``model``, and
        ``call_type``.

        Args:
            user_id: UUID of the user whose usage to retrieve.

        Returns:
            A list of dicts, each containing ``provider``, ``model``, and
            ``call_type`` from the most recent ``TokenUsage`` rows.
        """
        result = await self._db.execute(
            select(
                TokenUsage.provider,
                TokenUsage.model,
                TokenUsage.call_type,
            )
            .where(TokenUsage.user_id == user_id)
            .order_by(TokenUsage.created_at.desc())
            .limit(200)
        )
        return [dict(row._mapping) for row in result.fetchall()]
