"""Per-turn outcome telemetry.

Stores a small bounded ring of ``(intent, tool_sequence, repair_fired,
redteam_severity, duration_ms, clarification_asked)`` records on
``session.state["turn_telemetry"]``. Used later for policy learning,
quality dashboards, and debug audits. Pure DB write, no LLM, fire-
and-forget — never raises into the orchestrator.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm.attributes import flag_modified

from app.db.session import async_session_factory
from app.models.assistant import AssistantSession

log = logging.getLogger(__name__)

_TELEMETRY_KEY = "turn_telemetry"
_TELEMETRY_CAP = 50


async def record_turn_outcome(
    *,
    session_id: UUID,
    user_id: UUID,  # noqa: ARG001 — accepted for symmetry; not yet used
    intent_label: str,
    intent_confidence: float,
    complexity: str,
    tool_sequence: list[str],
    clarification_asked: bool,
    repair_fired: bool,
    redteam_severity: str | None,
    duration_ms: int,
    citation_count: int,
    grounded_paper_count: int,
) -> None:
    """Append one record to the session's telemetry ring."""
    try:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "intent": intent_label[:60],
            "confidence": round(float(intent_confidence or 0.0), 3),
            "complexity": complexity,
            "tools": [str(t)[:48] for t in (tool_sequence or [])][:12],
            "clarification": bool(clarification_asked),
            "repair": bool(repair_fired),
            "redteam_severity": (redteam_severity or "none")[:16],
            "duration_ms": int(duration_ms or 0),
            "citations": int(citation_count or 0),
            "grounded_papers": int(grounded_paper_count or 0),
        }
        async with async_session_factory() as db:
            row = await db.get(AssistantSession, session_id)
            if row is None:
                return
            state = dict(row.state or {})
            ring = list(state.get(_TELEMETRY_KEY) or [])
            ring.append(record)
            if len(ring) > _TELEMETRY_CAP:
                ring = ring[-_TELEMETRY_CAP:]
            state[_TELEMETRY_KEY] = ring
            row.state = state
            flag_modified(row, "state")
            await db.commit()
    except Exception as exc:
        log.debug("record_turn_outcome failed session=%s: %s", session_id, exc)
