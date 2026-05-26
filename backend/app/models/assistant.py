"""Persistent Research Assistant models."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class AssistantSessionStatus(str, enum.Enum):
    """Lifecycle state for a research assistant session."""

    active = "active"
    archived = "archived"


class AssistantMessageRole(str, enum.Enum):
    """Message author role."""

    user = "user"
    assistant = "assistant"
    system = "system"


class AssistantTaskStatus(str, enum.Enum):
    """Background task state surfaced in the assistant workspace."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AssistantStepStatus(str, enum.Enum):
    """Per-step lifecycle inside an orchestrated assistant turn."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    skipped = "skipped"


class AssistantSession(Base):
    """A persistent, branchable research investigation."""

    __tablename__ = "assistant_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    title: Mapped[str] = mapped_column(String(240), nullable=False, default="Untitled investigation")
    namespace_key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    topic_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    parent_session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_sessions.id", ondelete="SET NULL"), nullable=True
    )
    branch_from_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    orientation: Mapped[str] = mapped_column(String(30), default="both")
    expertise_level: Mapped[str] = mapped_column(String(30), default="practitioner")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[AssistantSessionStatus] = mapped_column(
        Enum(AssistantSessionStatus), default=AssistantSessionStatus.active
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["AssistantMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="AssistantMessage.created_at"
    )
    tasks: Mapped[list["AssistantTask"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="AssistantTask.created_at"
    )


class AssistantMessage(Base):
    """A persisted assistant workspace message with structured render payloads."""

    __tablename__ = "assistant_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_sessions.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    role: Mapped[AssistantMessageRole] = mapped_column(Enum(AssistantMessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    message_type: Mapped[str] = mapped_column(String(50), default="text")
    citations: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    artifact_refs: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["AssistantSession"] = relationship(back_populates="messages")


class AssistantTask(Base):
    """Persistent record for a long-running assistant orchestration task."""

    __tablename__ = "assistant_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_sessions.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    assistant_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    task_type: Mapped[str] = mapped_column(String(50), nullable=False, default="assistant")
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    namespace_key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[AssistantTaskStatus] = mapped_column(
        Enum(AssistantTaskStatus), default=AssistantTaskStatus.pending, index=True
    )

    progress: Mapped[dict] = mapped_column(JSONB, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped["AssistantSession"] = relationship(back_populates="tasks")


class AssistantStep(Base):
    """A single tool invocation inside an assistant turn.

    Steps are the unit of resumability, cancellation, and reasoning-tree
    rendering. The orchestrator writes one row per planned tool call;
    progress, output, and errors are captured here so a crashed worker can
    resume from the last completed step instead of re-running the whole turn.
    """

    __tablename__ = "assistant_steps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_sessions.id", ondelete="CASCADE"), index=True
    )
    parent_message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    parent_step_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_steps.id", ondelete="SET NULL"), nullable=True
    )
    job_id: Mapped[str] = mapped_column(String(100), index=True)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(240), nullable=False, default="")
    status: Mapped[AssistantStepStatus] = mapped_column(
        Enum(AssistantStepStatus), default=AssistantStepStatus.pending, index=True
    )

    input_params: Mapped[dict] = mapped_column(JSONB, default=dict)
    output: Mapped[dict] = mapped_column(JSONB, default=dict)
    progress: Mapped[dict] = mapped_column(JSONB, default=dict)
    cost: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AssistantAttachment(Base):
    """User-supplied content attached to an assistant session.

    Attachments are session-scoped: a note, URL, paper reference, or
    extracted document text the user dropped into the workspace. The
    assistant retrieves them alongside the global corpus when answering
    questions in the same session, but they are NEVER mixed into the
    user's main paper feed unless explicitly imported via arxiv_import.
    """

    __tablename__ = "assistant_attachments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_sessions.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)   # note | url | paper_ref | pdf | image
    label: Mapped[str] = mapped_column(String(240), default="")
    content: Mapped[str | None] = mapped_column(Text, nullable=True)            # text body / extracted markdown
    url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    paper_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AssistantArtifact(Base):
    """Generated outputs produced inside a session.

    Polymorphic registry pointing at the canonical row in its native table
    (study summary, idea capsule, podcast file, slide deck, mermaid blob,
    comparison report). The session view uses this as the unit of "things
    the user revisits, pins, exports."
    """

    __tablename__ = "assistant_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_sessions.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    producing_step_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_steps.id", ondelete="SET NULL"), nullable=True
    )
    producing_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    kind: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    ref_id: Mapped[str] = mapped_column(String(120), nullable=False)
    title: Mapped[str] = mapped_column(String(240), default="")
    href: Mapped[str | None] = mapped_column(String(500), nullable=True)
    preview: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MemoryRevision(Base):
    """Append-only audit trail for every long-term memory write.

    Each user-facing change to a tracked memory entry (auto-write,
    manual write, overwrite, delete, restore) lands here as one row so
    the user can inspect history, compare versions, and restore an
    earlier value. The live state still lives on
    :class:`AssistantSession.state` JSONB — this table is the *audit
    log*, never the source of truth for retrieval.

    Strict isolation: every row carries ``user_id`` directly and the
    inspect endpoints query by ``user_id`` first. A leaked session id
    from another user can never surface someone else's revision
    history.

    ``action`` is one of:
        * ``create``     — first time this key was written.
        * ``update``     — value changed; ``previous_value`` records
                           the prior value for compare/restore.
        * ``delete``     — the entry was removed.
        * ``restore``    — an earlier revision's value was reapplied.
        * ``supersede``  — auto-memory consolidation merged two entries.

    ``status`` reflects the entry's state AT THE TIME of this revision
    (active / stale / superseded / deleted). The current entry's
    status is recomputed from the most recent revision plus the live
    state (TTL freshness).
    """

    __tablename__ = "memory_revisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Root session that owns the memory bucket — for both medium (tree
    # memory) and long (namespace memory), the bucket sits on the
    # root session's state.
    root_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assistant_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    # Originating session — for medium/short writes via the memory
    # tool, this is the child session that triggered the write. For
    # long-tier writes via auto_memory it equals root_session_id. Kept
    # so the audit trail shows which conversation the entry came
    # from.
    origin_session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    tier: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # short / medium / long
    namespace_key: Mapped[str] = mapped_column(String(120), default="")
    # Subject / topic tags carried over from the namespace classifier so
    # the inspect API can filter by them without re-parsing
    # ``namespace_key`` on every read. Subject is the broad bucket
    # (e.g. ``cs``); topic is the fine-grained leaf (e.g. ``AI``).
    subject: Mapped[str] = mapped_column(String(60), default="", index=True)
    topic: Mapped[str] = mapped_column(String(60), default="", index=True)
    key: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, default="")
    previous_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    entry_type: Mapped[str] = mapped_column(String(40), default="context")
    source: Mapped[str] = mapped_column(String(40), default="manual")  # manual / auto / restore
    action: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default="active")  # active / stale / superseded / deleted
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    ttl_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extras: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True,
    )
