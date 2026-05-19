"""ArtifactRepository — CRUD for GeneratedArtifact rows.

Only this layer issues SQL for the generated_artifacts table. All callers
(workflows, API routes) must go through this class to preserve the
repository pattern boundary.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact import ArtifactStatus, GeneratedArtifact, GenerationType, SourceType

log = logging.getLogger(__name__)


class ArtifactRepository:
    """Database access layer for :class:`~app.models.artifact.GeneratedArtifact`.

    All methods are async and operate within the provided ``AsyncSession``.
    Callers are responsible for committing the session after writes.

    Args:
        db: Active SQLAlchemy async session.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the repository with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for all queries.
        """
        self._db = db

    # ── Create ─────────────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        user_id: UUID,
        generation_type: GenerationType,
        source_type: SourceType,
        source_id: UUID,
        expertise_level: str | None = None,
        orientation: str | None = None,
    ) -> GeneratedArtifact:
        """Insert a new artifact row in ``queued`` state and return it.

        Args:
            user_id: Owner of the artifact.
            generation_type: Type of media being generated.
            source_type: The kind of source entity.
            source_id: UUID of the source entity.
            expertise_level: User's expertise level at the time of request.
            orientation: User's orientation at the time of request.

        Returns:
            The newly flushed (not yet committed) ``GeneratedArtifact``.
        """
        artifact = GeneratedArtifact(
            user_id=user_id,
            generation_type=generation_type,
            source_type=source_type,
            source_id=source_id,
            status=ArtifactStatus.queued,
            expertise_level=expertise_level,
            orientation=orientation,
        )
        self._db.add(artifact)
        await self._db.flush()
        log.debug(
            "artifact.create id=%s type=%s source=%s/%s",
            artifact.id, generation_type, source_type, source_id,
        )
        return artifact

    # ── Read ───────────────────────────────────────────────────────────────────

    async def get_by_id(self, artifact_id: UUID) -> GeneratedArtifact | None:
        """Return a single artifact by primary key, or ``None`` if not found."""
        result = await self._db.execute(
            select(GeneratedArtifact).where(GeneratedArtifact.id == artifact_id)
        )
        return result.scalar_one_or_none()

    async def get_latest_completed(
        self,
        *,
        user_id: UUID,
        source_id: UUID,
        generation_type: GenerationType,
    ) -> GeneratedArtifact | None:
        """Return the most recent *completed* artifact for a (user, source, type) triple.

        Used to serve cached results and skip re-generation when valid
        artifacts already exist.

        Args:
            user_id: Owner to scope the lookup to.
            source_id: Source entity UUID.
            generation_type: Type of artifact to look for.

        Returns:
            Most recently completed ``GeneratedArtifact``, or ``None``.
        """
        result = await self._db.execute(
            select(GeneratedArtifact)
            .where(
                GeneratedArtifact.user_id == user_id,
                GeneratedArtifact.source_id == source_id,
                GeneratedArtifact.generation_type == generation_type,
                GeneratedArtifact.status == ArtifactStatus.completed,
            )
            .order_by(GeneratedArtifact.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def find_reusable_completed_global(
        self,
        *,
        source_id: UUID,
        generation_type: GenerationType,
        expertise_level: str | None,
        orientation: str | None,
        provider: str | None,
        model_used: str | None,
        parser_used: str | None,
    ) -> GeneratedArtifact | None:
        """Find a globally-reusable completed artifact across all users.

        Generation outputs are deterministic functions of the source paper/capsule
        and the (expertise, orientation, provider, model, parser) tuple — none of
        which is user-specific. When any user has already produced the matching
        artifact, every other user requesting the same combination should reuse
        the heavy outputs (blob, content, tokens) rather than pay for re-generation.

        Returns the **oldest** matching completed artifact, so the canonical row
        is stable (lookups don't bounce between competing copies as new users
        regenerate). Returns ``None`` if no global match exists.
        """
        stmt = select(GeneratedArtifact).where(
            GeneratedArtifact.source_id == source_id,
            GeneratedArtifact.generation_type == generation_type,
            GeneratedArtifact.status == ArtifactStatus.completed,
        )
        if expertise_level is not None:
            stmt = stmt.where(GeneratedArtifact.expertise_level == expertise_level)
        if orientation is not None:
            stmt = stmt.where(GeneratedArtifact.orientation == orientation)
        if provider is not None:
            stmt = stmt.where(GeneratedArtifact.provider == provider)
        if model_used is not None:
            stmt = stmt.where(GeneratedArtifact.model_used == model_used)
        # parser_used is permissive — only constrain when the requester knows the parser.
        if parser_used is not None:
            stmt = stmt.where(
                (GeneratedArtifact.parser_used == parser_used)
                | (GeneratedArtifact.parser_used.is_(None))
            )

        stmt = stmt.order_by(GeneratedArtifact.created_at.asc()).limit(1)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def count_references_to_blob(
        self,
        *,
        blob_path: str,
        exclude_artifact_id: UUID | None = None,
    ) -> int:
        """Count how many artifact rows reference a given blob path.

        Used to decide whether deleting a single artifact should also delete
        the underlying blob. With cross-user dedup, multiple artifacts may
        share the same ``blob_path``; the blob should survive until the last
        referencing row is removed.
        """
        from sqlalchemy import func as _func

        stmt = select(_func.count()).select_from(GeneratedArtifact).where(
            GeneratedArtifact.blob_path == blob_path,
        )
        if exclude_artifact_id is not None:
            stmt = stmt.where(GeneratedArtifact.id != exclude_artifact_id)
        result = await self._db.execute(stmt)
        return int(result.scalar_one() or 0)

    async def list_for_source(
        self,
        *,
        user_id: UUID,
        source_id: UUID,
    ) -> list[GeneratedArtifact]:
        """Return all artifacts for a source entity, newest first.

        Includes all statuses so the UI can show in-progress jobs alongside
        completed ones.

        Args:
            user_id: Owner filter.
            source_id: Source entity UUID.

        Returns:
            List of ``GeneratedArtifact`` ordered by ``created_at`` descending.
        """
        result = await self._db.execute(
            select(GeneratedArtifact)
            .where(
                GeneratedArtifact.user_id == user_id,
                GeneratedArtifact.source_id == source_id,
            )
            .order_by(GeneratedArtifact.created_at.desc())
        )
        return list(result.scalars().all())

    # ── Update ─────────────────────────────────────────────────────────────────

    async def mark_running(self, artifact_id: UUID) -> None:
        """Transition an artifact from ``queued`` to ``running``."""
        artifact = await self.get_by_id(artifact_id)
        if artifact and artifact.status != ArtifactStatus.failed:
            artifact.status = ArtifactStatus.running

    async def mark_completed(
        self,
        artifact_id: UUID,
        *,
        blob_path: str | None = None,
        content: dict | None = None,
        provider: str | None = None,
        model_used: str | None = None,
        parser_used: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """Transition artifact to ``completed`` and persist all output fields.

        Args:
            artifact_id: ID of the artifact to update.
            blob_path: BlobStorage key for binary artifacts (audio, HTML).
            content: JSON content for non-binary artifacts.
            provider: LLM provider used.
            model_used: Primary model identifier.
            parser_used: PDF parser used, if any.
            input_tokens: Aggregated input tokens across all LLM calls.
            output_tokens: Aggregated output tokens across all LLM calls.
            duration_ms: Total wall-clock time of the generation workflow.
            metadata: Additional provider-specific metadata.
        """
        artifact = await self.get_by_id(artifact_id)
        if not artifact:
            log.warning("artifact.mark_completed: id=%s not found", artifact_id)
            return
        if artifact.status == ArtifactStatus.failed:
            log.info("artifact.mark_completed: id=%s already failed/cancelled — skipping", artifact_id)
            return

        artifact.status = ArtifactStatus.completed
        artifact.completed_at = datetime.now(timezone.utc)

        if blob_path is not None:
            artifact.blob_path = blob_path
        if content is not None:
            artifact.content = content
        if provider is not None:
            artifact.provider = provider
        if model_used is not None:
            artifact.model_used = model_used
        if parser_used is not None:
            artifact.parser_used = parser_used
        if input_tokens:
            artifact.input_tokens = input_tokens
        if output_tokens:
            artifact.output_tokens = output_tokens
        if duration_ms:
            artifact.generation_duration_ms = duration_ms
        if metadata:
            artifact.artifact_metadata = metadata

        log.debug("artifact.completed id=%s type=%s", artifact_id, artifact.generation_type)

    async def mark_failed(self, artifact_id: UUID, *, error_message: str) -> None:
        """Transition artifact to ``failed`` and record the error message.

        Args:
            artifact_id: ID of the artifact to fail.
            error_message: Human-readable error description for the UI.
        """
        artifact = await self.get_by_id(artifact_id)
        if not artifact:
            return

        artifact.status = ArtifactStatus.failed
        artifact.completed_at = datetime.now(timezone.utc)
        artifact.error_message = error_message[:2000]  # guard against huge stack traces
        log.warning("artifact.failed id=%s err=%.200s", artifact_id, error_message)
