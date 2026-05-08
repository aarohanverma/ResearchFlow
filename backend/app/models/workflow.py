"""WorkflowRun — idempotency tracking for LangGraph workflows."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class WorkflowStatus(str, enum.Enum):
    """Execution state of a workflow run stored in ``WorkflowRun``."""

    running = "running"
    completed = "completed"
    failed = "failed"
    degraded = "degraded"


class WorkflowRun(Base):
    """One row per workflow execution. Prevents double-runs and supports audit."""

    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    workflow_name: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_key: Mapped[str] = mapped_column(String(200), nullable=False)  # namespace_key or user_id
    run_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    status: Mapped[WorkflowStatus] = mapped_column(
        Enum(WorkflowStatus), default=WorkflowStatus.running
    )
    error_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("workflow_name", "scope_key", "run_date", name="uq_workflow_run"),
    )


class TokenUsage(Base):
    """Per-call token accounting. Every LLM/embedding/image call writes a row."""

    __tablename__ = "token_usage"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)

    workflow: Mapped[str | None] = mapped_column(String(100), nullable=True)
    node: Mapped[str | None] = mapped_column(String(100), nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    call_type: Mapped[str] = mapped_column(String(30), nullable=False)  # llm | embedding | image

    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    latency_ms: Mapped[int] = mapped_column(default=0)
    success: Mapped[bool] = mapped_column(default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
