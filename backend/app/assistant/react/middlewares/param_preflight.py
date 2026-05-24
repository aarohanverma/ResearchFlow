"""Param preflight + auto-repair middleware.

The first line of defence against the failure mode that prompted this
work: the model emits ``params={}`` or ``{"query": "__to_fill__"}`` to
a retrieval tool that requires a concrete ``query``. Without
intervention, pydantic validation blows up with an opaque
``query field required`` and the model never recovers.

This middleware runs **before** every tool dispatch and:

  1. Strips known placeholder patterns from supplied params (so the
     downstream validator sees them as missing, not garbage).
  2. Auto-fills missing required fields from durable per-turn sources:
     the user's query for ``query`` / ``question`` / ``claim`` /
     ``topic``; the paper-ID ledger for ``paper_ids`` / ``paper_id``.
  3. Re-attempts pydantic validation if it failed on the model's
     params; on success, surfaces the auto-fill notes to the
     scratchpad so the model sees what got rewritten.

The actual repair logic lives in
:func:`app.assistant.react_loop._preflight_and_repair_params` — we
delegate so the middleware stays a thin wrapper around the existing
function. That keeps the existing tests for the repair function
authoritative.
"""

from __future__ import annotations

from typing import Any

from app.assistant.react.middleware import CONTINUE, DispatchOverride, PreDispatchResult
from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tools.registry import get_tool


class ParamPreflightMiddleware(BaseMiddleware):
    """Repair tool params before pydantic validation runs.

    Wraps :func:`app.assistant.react_loop._preflight_and_repair_params`
    so all the existing placeholder + auto-fill logic carries over
    untouched. The middleware records its repair notes onto the
    scratchpad so the model knows what changed.

    The hook only fires for real tools (skips ``finalize`` / ``critique``
    / ``fanout`` / ``subagent`` pseudo-actions — those have no schema).
    """

    name = "param_preflight"

    async def before_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
    ) -> PreDispatchResult:
        # Lazy import to avoid a cyclic dependency with react_loop.
        from app.assistant.react_loop import _preflight_and_repair_params

        tool = get_tool(action)
        if tool is None:
            # Unknown tool — let the loop's own "tool_not_found" branch
            # handle it; nothing we can repair without a schema.
            return CONTINUE
        try:
            schema_dict = tool.input_schema.model_json_schema()
        except Exception:  # noqa: BLE001 — test mocks may not be real models
            schema_dict = {}
        if not schema_dict:
            return CONTINUE

        repaired, notes = _preflight_and_repair_params(
            action,
            params if isinstance(params, dict) else {},
            schema_dict,
            query=state.query,
            ledger=state.ledger,
        )
        if not notes and repaired == params:
            return CONTINUE
        # Compact repair notice — surfaces what we rewrote so the
        # model can see its placeholder got fixed instead of silently
        # re-emitting it next iteration.
        note = (
            f"Auto-repaired params for {action}: " + "; ".join(notes)[:600]
            if notes else None
        )
        return DispatchOverride(
            action=None,
            params=repaired,
            note_for_scratchpad=note,
        )
