"""Memory write/recall/delete tools for the Research Assistant.

Three-tier memory system — all stored in AssistantSession.state (JSONB):

  short  — implicit: conversation history injected into every turn context.
  medium — session.state["memory"]: typed facts specific to this investigation.
            Inherited by branch sessions via parent context loading.
  long   — session.state["ns_memory"]: namespace-level insights that survive
            across sessions. Copied forward to new sessions in the same
            namespace by the session service so discoveries persist.

Each entry is stored as a dict with keys: value, type, ts.
Typed memory enables structured recall (filter by finding/preference/concept/hypothesis).

The planner calls memory_write to store facts, memory_recall to retrieve them,
and memory_delete to remove stale entries. The orchestrator always injects the
current memory into the planner prompt automatically (see orchestrator._load_context).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult
from app.models.assistant import AssistantSession

log = logging.getLogger(__name__)

_VALID_TYPES = {"finding", "preference", "concept", "hypothesis", "context", "paper_note"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_value(entry: object) -> str:
    """Normalize stored entry — supports both legacy str and new dict format."""
    if isinstance(entry, dict):
        return entry.get("value", "")
    return str(entry)


def _entry_type(entry: object) -> str:
    if isinstance(entry, dict):
        return entry.get("type", "context")
    return "context"


def _entry_ts(entry: object) -> str:
    if isinstance(entry, dict):
        return entry.get("ts", "")
    return ""


# ── Write ─────────────────────────────────────────────────────────────────────


class MemoryWriteInput(BaseModel):
    key: str = Field(
        min_length=1,
        max_length=120,
        description="Short identifier (snake_case, e.g. 'user_background', 'key_finding_attention').",
    )
    value: str = Field(
        min_length=1,
        max_length=2000,
        description="The fact or insight to remember. Plain text, one concept per entry.",
    )
    scope: str = Field(
        default="medium",
        pattern="^(medium|long)$",
        description="'medium' = this session; 'long' = persists across all sessions in this namespace.",
    )
    memory_type: str = Field(
        default="context",
        description=(
            "Type of memory: 'finding' (research discovery), 'preference' (user likes/dislikes), "
            "'concept' (definition/explanation worth keeping), 'hypothesis' (tracked hypothesis), "
            "'paper_note' (note about a specific paper), 'context' (general session context)."
        ),
    )


class MemoryWriteOutput(BaseModel):
    stored: bool
    scope: str
    key: str
    memory_type: str


class MemoryWriteTool:
    """Store a typed key-value fact into session or namespace memory."""

    name = "memory_write"
    summary = (
        "Persist a typed factual insight or user preference into research memory. "
        "Use scope='medium' for session-specific facts. "
        "Use scope='long' for lasting cross-session insights in this namespace. "
        "Typed categories: finding (research discovery), preference (user preference), "
        "concept (definition), hypothesis (tracked hypothesis), paper_note (paper note), context (general). "
        "Write only genuinely useful facts — one clear write per turn is better than many trivial ones."
    )
    cost_class = "cheap"
    side_effects = True
    cancellable = False
    streamable = False
    input_schema = MemoryWriteInput
    output_schema = MemoryWriteOutput

    async def run(self, ctx: ToolContext, params: MemoryWriteInput) -> ToolResult:
        mem_type = params.memory_type if params.memory_type in _VALID_TYPES else "context"
        await ctx.emit_progress(30, f"Storing {params.scope}-term [{mem_type}] memory: {params.key!r}")
        try:
            async with ctx.db.begin_nested():
                row = await ctx.db.get(AssistantSession, ctx.session_id)
                if row is None:
                    return ToolResult(
                        output={"stored": False, "scope": params.scope, "key": params.key, "memory_type": mem_type},
                        summary="session not found",
                    )
                state = dict(row.state or {})
                bucket = "ns_memory" if params.scope == "long" else "memory"
                mem = dict(state.get(bucket) or {})
                mem[params.key] = {"value": params.value, "type": mem_type, "ts": _now_iso()}
                state[bucket] = mem
                row.state = state
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(row, "state")
                await ctx.db.flush()
        except Exception as exc:
            log.warning("memory_write failed: %s", exc)
            return ToolResult(
                output={"stored": False, "scope": params.scope, "key": params.key, "memory_type": mem_type},
                summary=f"write failed: {exc}",
            )

        await ctx.emit_progress(100, f"Saved {params.scope}-term memory [{mem_type}]")
        return ToolResult(
            output={"stored": True, "scope": params.scope, "key": params.key, "memory_type": mem_type},
            summary=f"Stored {params.scope}-term [{mem_type}] memory: {params.key!r}",
        )


# ── Recall ─────────────────────────────────────────────────────────────────────


class MemoryRecallInput(BaseModel):
    namespace_key: str = Field(default="")
    query: str = Field(default="", max_length=500, description="Optional keyword filter on key or value.")
    memory_type: str = Field(
        default="",
        description="Optional type filter: 'finding', 'preference', 'concept', 'hypothesis', 'paper_note', 'context'.",
    )


class MemoryRecallOutput(BaseModel):
    medium: dict
    long: dict
    total_medium: int
    total_long: int


class MemoryRecallTool:
    """Surface stored memory for the current session and namespace."""

    name = "memory_recall"
    summary = (
        "Retrieve stored research memory: session-level (medium) and namespace-level (long). "
        "Optionally filter by keyword (query) or type (finding/preference/concept/hypothesis/paper_note/context). "
        "Use when: user asks about prior context, their background, what was discovered in previous sessions, "
        "or at the start of a continuation when personalization matters."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = False
    streamable = False
    input_schema = MemoryRecallInput
    output_schema = MemoryRecallOutput

    async def run(self, ctx: ToolContext, params: MemoryRecallInput) -> ToolResult:
        await ctx.emit_progress(50, "Recalling research memory")
        try:
            row = await ctx.db.get(AssistantSession, ctx.session_id)
            state = dict(row.state or {}) if row else {}
            medium = dict(state.get("memory") or {})
            long_mem = dict(state.get("ns_memory") or {})

            # Branch session: inherit parent medium memory (child overrides on conflict)
            if row and row.parent_session_id:
                parent = await ctx.db.get(AssistantSession, row.parent_session_id)
                if parent:
                    pstate = dict(parent.state or {})
                    parent_medium = dict(pstate.get("memory") or {})
                    medium = {**parent_medium, **medium}
                    if not long_mem:
                        long_mem = dict(pstate.get("ns_memory") or {})

            # Apply filters
            q = (params.query or "").lower()
            t = (params.memory_type or "").lower().strip()

            def _matches(k: str, entry: object) -> bool:
                if t and _entry_type(entry) != t:
                    return False
                if q:
                    return q in k.lower() or q in _entry_value(entry).lower()
                return True

            medium = {k: v for k, v in medium.items() if _matches(k, v)}
            long_mem = {k: v for k, v in long_mem.items() if _matches(k, v)}

            # Normalize to flat value strings for backward compat while preserving type/ts in output
            medium_out = {
                k: {"value": _entry_value(v), "type": _entry_type(v), "ts": _entry_ts(v)}
                for k, v in medium.items()
            }
            long_out = {
                k: {"value": _entry_value(v), "type": _entry_type(v), "ts": _entry_ts(v)}
                for k, v in long_mem.items()
            }
        except Exception as exc:
            log.warning("memory_recall failed: %s", exc)
            medium_out, long_out = {}, {}

        await ctx.emit_progress(100, f"Recalled {len(medium_out)} medium + {len(long_out)} long-term memories")
        return ToolResult(
            output={"medium": medium_out, "long": long_out, "total_medium": len(medium_out), "total_long": len(long_out)},
            summary=f"{len(medium_out)} session + {len(long_out)} namespace memories recalled",
        )


# ── Delete ─────────────────────────────────────────────────────────────────────


class MemoryDeleteInput(BaseModel):
    key: str = Field(min_length=1, max_length=120, description="Key of the memory entry to remove.")
    scope: str = Field(
        default="medium",
        pattern="^(medium|long)$",
        description="Which memory bucket to delete from.",
    )


class MemoryDeleteOutput(BaseModel):
    deleted: bool
    scope: str
    key: str


class MemoryDeleteTool:
    """Remove a stale or incorrect memory entry."""

    name = "memory_delete"
    summary = (
        "Delete a specific memory entry by key. Use when a stored fact is outdated, "
        "incorrect, or no longer relevant. Provide the exact key and scope used when the entry was written."
    )
    cost_class = "cheap"
    side_effects = True
    cancellable = False
    streamable = False
    input_schema = MemoryDeleteInput
    output_schema = MemoryDeleteOutput

    async def run(self, ctx: ToolContext, params: MemoryDeleteInput) -> ToolResult:
        await ctx.emit_progress(50, f"Deleting {params.scope}-term memory: {params.key!r}")
        deleted = False
        try:
            async with ctx.db.begin_nested():
                row = await ctx.db.get(AssistantSession, ctx.session_id)
                if row:
                    state = dict(row.state or {})
                    bucket = "ns_memory" if params.scope == "long" else "memory"
                    mem = dict(state.get(bucket) or {})
                    if params.key in mem:
                        del mem[params.key]
                        state[bucket] = mem
                        row.state = state
                        from sqlalchemy.orm.attributes import flag_modified
                        flag_modified(row, "state")
                        await ctx.db.flush()
                        deleted = True
        except Exception as exc:
            log.warning("memory_delete failed: %s", exc)

        await ctx.emit_progress(100, "Memory entry removed" if deleted else "Key not found")
        return ToolResult(
            output={"deleted": deleted, "scope": params.scope, "key": params.key},
            summary=f"Deleted {params.scope}-term memory: {params.key!r}" if deleted else f"Key not found: {params.key!r}",
        )


memory_write_tool = MemoryWriteTool()
memory_recall_tool = MemoryRecallTool()
memory_delete_tool = MemoryDeleteTool()
