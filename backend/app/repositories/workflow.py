"""WorkflowRepository — idempotency and token usage tracking."""

from datetime import date, datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workflow import TokenUsage, WorkflowRun, WorkflowStatus


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
        """Return ``True`` if no successfully completed run exists for today.

        Checks for a ``WorkflowRun`` row with the given name and scope that has
        status ``completed`` and a ``run_date`` equal to today (or the supplied
        ``run_date``). Used as an idempotency guard so jobs do not repeat within
        the same calendar day.

        Args:
            workflow_name: Logical name of the workflow (e.g. ``"ingestion"``).
            scope_key: Scoping key, typically a namespace key (e.g. ``"cs.AI"``).
            run_date: The date to check against. Defaults to today in UTC when
                ``None``.

        Returns:
            ``True`` if no completed run exists for the given date, meaning the
            workflow should execute. ``False`` if it has already completed.
        """
        run_dt = self._utc_midnight(run_date)
        result = await self._db.execute(
            select(WorkflowRun).where(
                WorkflowRun.workflow_name == workflow_name,
                WorkflowRun.scope_key == scope_key,
                WorkflowRun.run_date == run_dt,
                WorkflowRun.status == WorkflowStatus.completed,
            )
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
