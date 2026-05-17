"""Tool contract for the Research Assistant orchestrator.

Each tool wraps an existing platform capability with a uniform interface so
the planner can reason about all capabilities through schemas alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession


CostClass = Literal["cheap", "moderate", "heavy"]
ProgressEmitter = Callable[[int, str], Awaitable[None]]
CancelChecker = Callable[[], Awaitable[bool]]


@dataclass
class ToolContext:
    """Per-tool-call execution context.

    Carries everything a tool needs without requiring it to know about the
    orchestrator or the database session lifecycle. The orchestrator builds
    a fresh context per step.
    """

    user_id: UUID
    session_id: UUID
    namespace_key: str
    namespace_keys: list[str]
    orientation: str
    expertise_level: str
    job_id: str
    parent_message_id: UUID
    db: AsyncSession
    should_cancel: CancelChecker
    emit_progress: ProgressEmitter
    parent_step_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Structured tool output handed back to the orchestrator.

    ``output`` is persisted to AssistantStep.output as-is (must be JSON-safe).
    ``artifacts`` produces AssistantArtifact rows. ``citations`` extends the
    parent message's citation list. ``cost`` (tokens, latency_ms, usd) feeds
    the standard TokenUsage accounting.
    """

    output: dict[str, Any]
    summary: str = ""
    citations: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AssistantTool(Protocol):
    """Protocol every assistant-orchestratable capability must satisfy."""

    name: str
    summary: str
    cost_class: CostClass
    side_effects: bool
    cancellable: bool
    streamable: bool
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]

    async def run(self, ctx: ToolContext, params: BaseModel) -> ToolResult: ...
