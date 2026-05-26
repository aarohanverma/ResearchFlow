"""HITL gate middleware — soft preview + ack for high-impact tools.

Today the gate fires only for ``genie_synthesize`` (creating a new
IdeaCapsule is a user-visible side effect that the user might want to
inspect / adjust before kicking off). Adding more gated tools is a
one-line addition to ``_GATED_TOOLS``.

Behavior on ``before_tool``:

1. Emit a ``react_hitl_pending`` SSE event with a compact preview
   payload (tool name, request_id, intent text, top seed papers).
2. Register a slot in the in-process inbox (see
   :mod:`app.assistant.hitl_inbox`) and ``await`` the user's
   decision with a short wall-clock cap (``_ACK_WINDOW_SEC``).
3. Apply the decision:

   * **approve** / **modify** → emit ``react_hitl_resolved`` and
     return a :class:`DispatchOverride` carrying the (possibly
     edited) params. The loop runs the tool as planned.
   * **skip** → emit ``react_hitl_resolved`` and return
     :class:`AbortDispatch` — the dispatch is dropped and the
     scratchpad records the user's veto. The loop continues on
     the next iteration so the model can finalize or pick a
     different action.
   * **timeout** → emit ``react_hitl_timeout`` and ``CONTINUE`` —
     the original params run through with a scratchpad note that
     the user was offered the gate and didn't engage. The note
     is important for the synthesizer so it can mention that the
     Genie capsule was created automatically without explicit
     user sign-off.

Why soft (not hard) suspend:

The user picked soft preview + ack. A research turn must always
converge: a hard suspend that waits forever would tie up server
resources, leave the UI showing "thinking…" indefinitely, and become
the next bug report ("the agent never finished"). The short ack
window gives the human a real opportunity to intervene while keeping
the agent's tail latency bounded.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.assistant.hitl_inbox import (
    HitlDecision,
    discard as inbox_discard,
    register_pending,
)
from app.assistant.react.middleware import (
    AbortDispatch,
    CONTINUE,
    DispatchOverride,
    PreDispatchResult,
)
from app.assistant.react.middlewares.base import BaseMiddleware

log = logging.getLogger(__name__)


# Tools that require a preview-and-ack pass before dispatch. Keep
# this list short — every gated tool taxes the user with a UI prompt,
# so we only gate calls that mint a durable side-effecting artifact.
_GATED_TOOLS: frozenset[str] = frozenset({"genie_synthesize"})

# How long the loop waits for a user decision before proceeding with
# the originally-emitted params. Short on purpose: a longer window
# starves the agent's tail latency, and the user can still react
# after-the-fact (the IdeaCapsule lands on the Genie page either way,
# so a missed approval is recoverable).
_ACK_WINDOW_SEC = 10.0


class HitlGateMiddleware(BaseMiddleware):
    """Soft preview + ack gate for high-impact tool dispatches."""

    name = "hitl_gate"

    async def before_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
    ) -> PreDispatchResult:
        if action not in _GATED_TOOLS:
            return CONTINUE

        # Build a compact preview the UI can render into a card
        # without making the SSE payload heavy. We carry the top seed
        # paper IDs + titles from the ledger so the user sees what
        # the loop intends to synthesize over.
        preview = self._build_preview(state, action, params)

        # Resolve session/user ids from the ToolContext. When either
        # is missing we degrade to CONTINUE — the gate is advisory,
        # so a context without owner info should not block the loop.
        ctx = getattr(state, "ctx", None)
        if ctx is None or getattr(ctx, "session_id", None) is None or getattr(ctx, "user_id", None) is None:
            return CONTINUE

        try:
            record = await register_pending(
                session_id=str(ctx.session_id),
                user_id=str(ctx.user_id),
                tool=action,
                params=params,
                preview=preview,
            )
        except Exception as exc:  # noqa: BLE001 — inbox failure must never abort the loop
            log.debug("hitl_gate: register_pending failed: %s", exc)
            return CONTINUE

        # Emit the SSE event so the frontend can render the ack card.
        # ``state.publish_event`` is the orchestrator's safe wrapper —
        # any failure swallowed there.
        state.publish_event(
            "react_hitl_pending",
            {
                "request_id": record.request_id,
                "tool": action,
                "preview": preview,
                "ack_window_sec": _ACK_WINDOW_SEC,
            },
        )
        state.pad.think(
            f"HITL gate: pausing up to {int(_ACK_WINDOW_SEC)}s for user approval "
            f"on {action}. Will proceed automatically on timeout."
        )

        # Wait — but never longer than the ack window. asyncio.wait_for
        # propagates CancelledError so the loop's cancel path still
        # works while we're paused at the gate.
        #
        # The outer try/finally guarantees ``inbox_discard`` runs on
        # ANY exit path other than a successful ``resolve()`` (which
        # pops the slot itself). Without this guard a propagating
        # exception between the await and the return — e.g. a
        # ``publish_event`` raise that slipped past the safe wrapper,
        # or a future-set_result corruption — would leave the slot
        # lingering in the inbox until the hard cap evicted it. The
        # ``_discarded`` flag stops us from double-discarding on the
        # already-handled timeout / cancel branches.
        decision: HitlDecision | None = None
        _discarded = False
        try:
            try:
                decision = await asyncio.wait_for(
                    record.future, timeout=_ACK_WINDOW_SEC,
                )
            except asyncio.TimeoutError:
                inbox_discard(record.request_id)
                _discarded = True
                state.publish_event(
                    "react_hitl_timeout",
                    {"request_id": record.request_id, "tool": action},
                )
                state.pad.think(
                    "HITL gate: user did not respond within the ack window — "
                    "proceeding with the originally-emitted params. The "
                    "synthesized idea will still surface on the Genie page; "
                    "the user can dismiss it there if it was unintended."
                )
                return CONTINUE
            except asyncio.CancelledError:
                inbox_discard(record.request_id)
                _discarded = True
                raise

            # Resolved by the user — fan out to the three concrete branches.
            state.publish_event(
                "react_hitl_resolved",
                {
                    "request_id": record.request_id,
                    "tool": action,
                    "status": decision.status,
                },
            )

            if decision.status == "skip":
                state.pad.think(
                    "HITL gate: user vetoed the Genie synthesis call. "
                    "Dropping the dispatch and continuing the loop."
                )
                return AbortDispatch(
                    reason="hitl_user_skip",
                    observation_summary=(
                        "User skipped the Genie synthesis at the approval gate. "
                        "No capsule was created. Consider whether a different "
                        "tool would address the user's question, or finalize "
                        "with the evidence already gathered."
                    ),
                    error="hitl_skip",
                )

            # approve / modify: use the user's params when supplied, else
            # fall back to the originally-emitted dict. ``modify`` is the
            # interesting branch — the user can adjust seed paper ids /
            # intent before we kick off the heavy workflow.
            new_params = dict(params)
            if decision.params:
                new_params.update(decision.params)
                state.pad.think(
                    "HITL gate: user approved with edits — using the modified "
                    f"params for {action}."
                )
            else:
                state.pad.think(
                    f"HITL gate: user approved {action} — proceeding."
                )
            return DispatchOverride(params=new_params)
        finally:
            # Catch-all: if we left the gate without going through one
            # of the discard-handled branches above (unexpected raise
            # mid-resolve-handling), make sure the slot doesn't linger.
            # ``inbox_discard`` is idempotent so the double-call on
            # the timeout/cancel paths is safe to skip via the flag.
            if not _discarded:
                # When a decision arrived, ``resolve()`` already
                # popped the slot — discard is a no-op. When the
                # success path raised after the await, this is the
                # only place we clean up.
                inbox_discard(record.request_id)

    # ── Preview builders ────────────────────────────────────────────

    def _build_preview(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Compose the UI preview payload for a gated dispatch.

        Keep it small — this rides on an SSE event and gets rendered
        into a card. We surface enough for the user to decide
        intelligently without dumping the full ledger / scratchpad.
        """
        if action == "genie_synthesize":
            return self._build_genie_preview(state, params)
        # Generic fallback for future gated tools.
        return {
            "tool": action,
            "summary": f"About to invoke {action}",
            "params": params,
        }

    def _build_genie_preview(
        self,
        state: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Tailored preview for ``genie_synthesize``.

        Surfaces the seed paper ids the loop is about to synthesize
        over, with titles pulled from the paper ledger so the user
        sees "Synthesize an idea from these 4 papers: X, Y, Z, …"
        rather than an opaque list of UUIDs.
        """
        seed_ids = list((params or {}).get("paper_ids") or [])
        ledger = getattr(state, "ledger", None)
        seeds: list[dict] = []
        if ledger is not None:
            for pid in seed_ids[:8]:
                info = ledger.by_id.get(str(pid)) or {}
                seeds.append({
                    "paper_id": str(pid),
                    "title": (info.get("title") or "")[:160],
                    "namespace": info.get("ns") or "",
                })
        intent = (params or {}).get("query") or ""
        return {
            "tool": "genie_synthesize",
            "summary": (
                "RA wants to synthesize a new research idea from these "
                f"{len(seed_ids)} papers. Approve to run, modify to edit "
                "the seed set or intent, or skip to drop the call."
            ),
            "intent": str(intent)[:400],
            "seeds": seeds,
            "seed_count": len(seed_ids),
        }
