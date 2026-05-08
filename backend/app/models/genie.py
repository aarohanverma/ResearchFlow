"""Genie idea synthesis models — Element Library and Idea Capsules."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class ElementType(str, enum.Enum):
    """Type of item dragged into the Genie synthesis cauldron."""

    concept = "concept"
    method = "method"
    paper = "paper"
    idea = "idea"   # previously synthesized Idea Capsule


class GenieElement(Base):
    """An item in the user's Genie element library — dragged into the cauldron."""

    __tablename__ = "genie_elements"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    element_type: Mapped[ElementType] = mapped_column(Enum(ElementType), nullable=False)
    label: Mapped[str] = mapped_column(String(500), nullable=False)

    # Source pointers — at most one set
    paper_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    knowledge_node_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    idea_capsule_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IdeaCapsule(Base):
    """The synthesized output of a Genie session — a grounded, testable hypothesis."""

    __tablename__ = "idea_capsules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    title: Mapped[str] = mapped_column(Text, nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    mechanism: Mapped[str | None] = mapped_column(Text, nullable=True)
    predicted_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    experimental_design: Mapped[str | None] = mapped_column(Text, nullable=True)
    anti_finding: Mapped[str | None] = mapped_column(Text, nullable=True)
    risks_and_limitations: Mapped[str | None] = mapped_column(Text, nullable=True)
    open_questions: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Grounding
    citation_paper_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    impact_chunk_ids: Mapped[list[str]] = mapped_column("grounding_chunk_ids", JSONB, default=list)

    novelty_score: Mapped[float] = mapped_column(Float, default=0.0)
    feasibility_score: Mapped[float] = mapped_column(Float, default=0.0)
    impact_score: Mapped[float] = mapped_column("grounding_score", Float, default=0.0)

    # Diagrams and code
    diagrams: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    poc_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    hero_image_blob_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Source elements used to generate this capsule
    seed_element_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)

    # Discovery mode
    is_scout_generated: Mapped[bool] = mapped_column(Boolean, default=False)

    # Origin tag: "manual" | "auto" | "query"
    source_mode: Mapped[str] = mapped_column(String(20), server_default="manual")
    # For query mode: the natural-language query the user typed
    source_query: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(30), default="draft")  # draft | saved | dismissed

    deep_dive_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    deep_dive_status: Mapped[str] = mapped_column(String(30), server_default="none")

    model_used: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship()  # noqa: F821


class GenieSession(Base):
    """Tracks a single Genie synthesis request for audit and streaming continuity."""

    __tablename__ = "genie_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    seed_element_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending|running|done|failed
    result_capsule_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
