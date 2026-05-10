"""GeneratedArtifact ORM model — server-persisted outputs for media generation.

Covers podcast (audio MP3) and slides (HTML/Marp). One row per generation
request; status progresses queued → running → completed | failed.

Designed for cloud parity: blob_path points into whichever BlobStorage
backend is active (local file, Azure Blob, S3), and content stores
structured JSON for format types that don't need a binary file.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class GenerationType(str, enum.Enum):
    """Active media artifact kinds.

    Two legacy enum members (``_legacy_a``, ``_legacy_b``) are retained as
    storage-level compatibility shims so any pre-existing DB rows from
    older deployments still deserialise without raising. They are filtered
    out of every API response and cannot be triggered.
    """

    podcast = "podcast"
    slides = "slides"
    _legacy_a = "video"          # deprecated — filtered at the API layer
    _legacy_b = "interactive"    # deprecated — filtered at the API layer


class SourceType(str, enum.Enum):
    """The entity the artifact was generated from."""

    paper = "paper"
    capsule = "capsule"
    folder = "folder"


class ArtifactStatus(str, enum.Enum):
    """Lifecycle state of a generation job."""

    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class GeneratedArtifact(Base):
    """One row per media generation request.

    Attributes:
        id: Primary key.
        user_id: Owner — cascades on user deletion.
        generation_type: podcast | slides.
        source_type: paper | capsule | folder — the entity being explained.
        source_id: UUID of the source entity (paper, capsule, or folder).
        status: queued → running → completed | failed.
        blob_path: BlobStorage key for binary artifacts (MP3, HTML). ``None``
            when no binary asset is produced.
        content: Structured JSON payload for non-binary types.
        expertise_level: newcomer | practitioner | expert — baked into generation.
        orientation: research | production | both — baked into generation.
        provider: LLM provider identifier (e.g. ``"openai"``).
        model_used: Specific model string used for the primary generation call.
        parser_used: PDF parser identifier if a PDF was parsed.
        input_tokens: Estimated input tokens consumed across all LLM calls.
        output_tokens: Estimated output tokens consumed across all LLM calls.
        generation_duration_ms: Wall-clock milliseconds for the full workflow.
        error_message: Stored on failure for UI display.
        artifact_metadata: Free-form JSONB for provider-specific extras.
        created_at: UTC timestamp of initial request.
        completed_at: UTC timestamp of completion or failure.
    """

    __tablename__ = "generated_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    generation_type: Mapped[GenerationType] = mapped_column(
        Enum(GenerationType), nullable=False, index=True
    )
    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType), nullable=False
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    status: Mapped[ArtifactStatus] = mapped_column(
        Enum(ArtifactStatus), nullable=False, default=ArtifactStatus.queued
    )

    # Storage — mutually exclusive by generation type
    blob_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Generation context — baked into the artifact at creation time
    expertise_level: Mapped[str | None] = mapped_column(String(50), nullable=True)
    orientation: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Provenance
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parser_used: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Token accounting (aggregated across all nodes in the workflow)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    generation_duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    artifact_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
