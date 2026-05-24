"""In-process inbox for Human-In-The-Loop ack/skip/modify decisions.

The ReAct loop uses this to pause briefly when it is about to invoke a
high-impact tool (today: ``genie_synthesize``) and give the user a
short window to:

* **approve** — proceed with the tool's intended params,
* **skip** — abort the dispatch entirely (the loop records the skip
  on the scratchpad and moves on),
* **modify** — proceed with an edited param dict (e.g. a different
  selection of seed papers).

Design constraints:

* **Soft, not hard** — the loop never blocks for longer than the
  configured ack window. A user who doesn't respond gets a
  "proceeding without explicit approval" scratchpad note and the
  loop continues with the originally-emitted params. This is
  deliberate: the gate is a courtesy, not a hard interrupt, so the
  research turn always converges.
* **In-process only** — the inbox is a process-local
  ``WeakValueDictionary`` of ``asyncio.Future`` slots. Multi-worker
  deployments would need a Redis-backed shim (the FastAPI worker
  that received the user's POST may not be the worker hosting the
  loop), so the gate transparently degrades to "timeout, proceed"
  in those setups. The current ResearchFlow local profile runs a
  single worker, so this is adequate; the upgrade path is to swap
  the registry for an async Redis pub/sub channel keyed by
  ``request_id`` without touching callers.

Public surface:

* :func:`register_pending` — middleware call. Returns a
  ``(request_id, future)`` pair the middleware awaits.
* :func:`resolve` — called by the API endpoint when the user
  responds. Resolves the future with the user's decision.
* :func:`peek` — non-blocking inspection; used by the API endpoint
  to validate that a request_id is still open before accepting a
  decision (so a stale tab doesn't post into an expired slot).
"""

from __future__ import annotations

import asyncio
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ── Public payload shapes ────────────────────────────────────────────────────


@dataclass
class HitlPendingRecord:
    """Server-side view of a pending HITL slot.

    Holds the metadata the API endpoint needs to validate an
    incoming decision (which user / session is this slot for, what
    tool was the loop about to invoke, what were the proposed params)
    and the future the middleware is awaiting on.

    The ``preview`` dict is the same payload that gets emitted on the
    ``react_hitl_pending`` SSE event, kept here so a late-loading UI
    that polls the inbox can render the same card even if it missed
    the SSE event.
    """

    request_id: str
    session_id: str
    user_id: str
    tool: str
    params: dict[str, Any]
    preview: dict[str, Any]
    future: asyncio.Future = field(repr=False)


@dataclass(frozen=True)
class HitlDecision:
    """What the user (or the timeout) decided.

    ``status`` is one of:

    * ``approve`` — proceed with ``params`` (defaults to the original
      params if the user didn't edit them),
    * ``skip``    — abort the dispatch; the middleware returns an
      ``AbortDispatch`` to the chain,
    * ``modify``  — proceed with the supplied ``params`` overriding
      the original (functionally identical to ``approve`` but kept
      separate so telemetry can distinguish blind-acks from edits),
    * ``timeout`` — the ack window expired without user input; the
      middleware proceeds with the original params and logs a
      scratchpad note so the synth knows the user was offered the
      gate and didn't respond.
    """

    status: str  # approve | skip | modify | timeout
    params: dict[str, Any] | None = None
    note: str = ""


# ── Registry ─────────────────────────────────────────────────────────────────


# Process-local registry. Keyed by request_id so a stale POST from a
# closed tab can be rejected cleanly. A WeakValueDictionary would let
# slots vanish under us once the awaiting middleware drops its
# reference, which would break the API endpoint's ability to validate
# the slot; we use a plain dict and explicitly delete on resolve /
# expire instead. The dict is bounded by the number of in-flight
# ReAct turns (typically ≤ a handful per worker), so unbounded growth
# isn't a realistic concern, but we guard against the pathological
# case in :func:`register_pending`.
_INBOX: dict[str, HitlPendingRecord] = {}
_INBOX_MUTEX = asyncio.Lock()
_INBOX_HARD_CAP = 256


async def register_pending(
    *,
    session_id: str,
    user_id: str,
    tool: str,
    params: dict[str, Any],
    preview: dict[str, Any],
) -> HitlPendingRecord:
    """Create a fresh pending slot and return the record.

    The middleware awaits ``record.future``; the API endpoint resolves
    it via :func:`resolve`. ``preview`` is opaque to the inbox itself
    — it carries the SSE payload the UI renders into a card.
    """
    async with _INBOX_MUTEX:
        if len(_INBOX) >= _INBOX_HARD_CAP:
            # Drop the oldest slot if we somehow accumulate too many
            # un-resolved requests. This is a safety net for the
            # pathological "every ReAct turn left its HITL request
            # un-resolved" case — in practice the timeout path
            # always cleans up.
            oldest = next(iter(_INBOX))
            stale = _INBOX.pop(oldest, None)
            if stale and not stale.future.done():
                stale.future.set_result(HitlDecision(status="timeout", note="evicted"))
            log.warning("hitl_inbox: hard cap reached, evicted request_id=%s", oldest)
        request_id = _uuid.uuid4().hex
        # ``get_event_loop()`` is deprecated when no loop is running;
        # this function is only reachable from an awaiting middleware,
        # so the running loop is the authoritative one to bind the
        # future to. Falling back to ``get_event_loop`` would risk
        # creating a fresh loop here (Python 3.10+ behaviour) that the
        # middleware's ``await`` would never see resolve.
        loop = asyncio.get_running_loop()
        record = HitlPendingRecord(
            request_id=request_id,
            session_id=str(session_id),
            user_id=str(user_id),
            tool=tool,
            params=dict(params or {}),
            preview=dict(preview or {}),
            future=loop.create_future(),
        )
        _INBOX[request_id] = record
        return record


def peek(request_id: str) -> HitlPendingRecord | None:
    """Non-blocking read; returns None if the slot is unknown or
    already resolved."""
    rec = _INBOX.get(request_id)
    if rec is None:
        return None
    if rec.future.done():
        return None
    return rec


def resolve(
    *,
    request_id: str,
    session_id: str,
    user_id: str,
    decision: HitlDecision,
) -> bool:
    """Resolve a pending slot with the user's decision.

    Validates ownership before resolving — a request_id alone isn't
    enough; both ``session_id`` and ``user_id`` must match what was
    registered so a leaked id can't be replayed from a different
    session. Returns ``True`` when the slot was resolved, ``False``
    when the slot is unknown, already resolved, or owned by a
    different (session, user). Caller (API endpoint) treats False as
    a 404 — the slot has either expired or never existed.
    """
    rec = _INBOX.get(request_id)
    if rec is None:
        return False
    if rec.session_id != str(session_id) or rec.user_id != str(user_id):
        # Ownership mismatch — refuse silently. Logging at warn so
        # we'd see a misuse pattern (cross-session id leak) without
        # leaking the offending values into the API response.
        log.warning(
            "hitl_inbox.resolve: ownership mismatch request_id=%s expected_session=%s actual=%s",
            request_id, rec.session_id, session_id,
        )
        return False
    if rec.future.done():
        return False
    rec.future.set_result(decision)
    # Eager cleanup — the middleware no longer needs the slot once
    # resolved, and keeping it around invites stale-id-replay bugs.
    _INBOX.pop(request_id, None)
    return True


def discard(request_id: str) -> None:
    """Best-effort cleanup — called by the middleware after the
    timeout path so the slot doesn't linger in the registry.

    Safe to call on an already-removed slot; the inbox is a soft
    advisory mechanism, not a transactional one.
    """
    _INBOX.pop(request_id, None)


__all__ = [
    "HitlDecision",
    "HitlPendingRecord",
    "discard",
    "peek",
    "register_pending",
    "resolve",
]
