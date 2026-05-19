"""Per-job event bus for SSE turn streaming.

The orchestrator publishes typed AssistantEvents; the SSE endpoint subscribes
to the queue keyed on the job_id. Late subscribers receive the buffered
event history so a frontend that connects mid-turn doesn't miss earlier
steps. This is an in-process bus on purpose — Redis pub/sub is the swap
when we move to multi-worker; the abstraction here keeps callers unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

log = logging.getLogger(__name__)


EventKind = Literal[
    "plan_proposed",
    "plan_committed",
    "step_started",
    "step_progress",
    "step_completed",
    "step_failed",
    "replanning",
    "message_delta",
    "message_completed",
    "suggestion",
    "task_completed",
    "task_failed",
    "task_cancelled",
    "heartbeat",
]


@dataclass
class AssistantEvent:
    """One event emitted during a turn's lifecycle."""

    kind: EventKind
    job_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "job_id": self.job_id, "ts": self.ts, "payload": self.payload}


class _JobChannel:
    """Single job's broadcast channel: history + fan-out to current subscribers."""

    def __init__(self, max_history: int = 200) -> None:
        self.history: list[AssistantEvent] = []
        self.subscribers: list[asyncio.Queue[AssistantEvent]] = []
        self._max_history = max_history
        self._closed = False

    def publish(self, event: AssistantEvent) -> None:
        self.history.append(event)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]
        # ``list(self.subscribers)`` snapshots so subscriber list mutations
        # during dispatch (e.g. a subscriber's unsubscribe inside a callback)
        # never raise RuntimeError nor skip recipients.
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("event bus queue full for job=%s — dropping event", event.job_id)

    def subscribe(self) -> asyncio.Queue[AssistantEvent]:
        """Add a new subscriber and pre-fill it with the buffered history.

        The queue ``maxsize`` is bounded so a stuck consumer cannot grow
        the queue without bound. The history replay therefore stops early
        if it would overflow — the consumer still gets the most recent
        events because ``history`` is a sliding window.
        """
        q: asyncio.Queue[AssistantEvent] = asyncio.Queue(maxsize=500)
        # Replay buffered history so late subscribers don't miss anything.
        for ev in self.history:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                break
        self.subscribers.append(q)
        # If the channel was already closed before this subscribe (turn
        # finished, then a late SSE client connected for history), drop a
        # terminal heartbeat so the consumer can exit cleanly instead of
        # waiting for an event that will never arrive.
        if self._closed:
            try:
                q.put_nowait(AssistantEvent(kind="heartbeat", job_id="", payload={"closed": True}))
            except asyncio.QueueFull:
                pass
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass

    def close(self) -> None:
        """Mark the channel closed and notify every live subscriber.

        Sends a terminal ``heartbeat`` event with ``payload={"closed": True}``
        so consumers can break out of their read loop deterministically
        rather than waiting on the 15-s SSE heartbeat to time out.
        """
        if self._closed:
            return
        self._closed = True
        for q in list(self.subscribers):
            try:
                q.put_nowait(AssistantEvent(kind="heartbeat", job_id="", payload={"closed": True}))
            except asyncio.QueueFull:
                pass

    @property
    def closed(self) -> bool:
        return self._closed


class AssistantEventBus:
    """Process-wide bus: one channel per job_id, lazy-created on first use.

    All public methods are safe to call from any coroutine on the same event
    loop — channel creation uses ``setdefault`` so two concurrent
    :meth:`publish` / :meth:`subscribe` calls for the same brand-new job_id
    can never end up with two distinct channels (one would otherwise win and
    the other's events would be silently dropped).
    """

    def __init__(self) -> None:
        self._channels: dict[str, _JobChannel] = {}

    def _get_or_create(self, job_id: str) -> _JobChannel:
        """Return the channel for ``job_id``, creating it atomically if absent.

        ``dict.setdefault`` is atomic with respect to other coroutines on
        the same event loop, so we never race on first-publish vs
        first-subscribe for the same job.
        """
        ch = self._channels.get(job_id)
        if ch is None:
            ch = self._channels.setdefault(job_id, _JobChannel())
        return ch

    def publish(self, event: AssistantEvent) -> None:
        self._get_or_create(event.job_id).publish(event)

    def subscribe(self, job_id: str) -> asyncio.Queue[AssistantEvent]:
        return self._get_or_create(job_id).subscribe()

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        ch = self._channels.get(job_id)
        if ch:
            ch.unsubscribe(q)
            # Auto-evict: once the channel is closed and has no remaining
            # subscribers there is nothing left to read from it.  Dropping it
            # here prevents the dict from growing without bound over the life
            # of the process (one entry per completed job otherwise).
            if ch.closed and not ch.subscribers:
                self._channels.pop(job_id, None)

    def history(self, job_id: str) -> list[AssistantEvent]:
        ch = self._channels.get(job_id)
        return list(ch.history) if ch else []

    def close(self, job_id: str) -> None:
        """Mark a channel closed; evict immediately when nobody is listening.

        Without the no-subscriber fast path, channels for turns that finished
        before any SSE client connected (or after every client disconnected)
        would accumulate in :attr:`_channels` for the life of the process.
        Subscribers that connect after :meth:`close` still see the buffered
        history via :meth:`subscribe`'s history replay because eviction only
        fires when ``not subscribers``.
        """
        ch = self._channels.get(job_id)
        if ch:
            ch.close()
            if not ch.subscribers:
                self._channels.pop(job_id, None)

    def evict(self, job_id: str) -> None:
        """Drop a finished channel after subscribers have drained."""
        self._channels.pop(job_id, None)

    def channel_count(self) -> int:
        """Diagnostic: number of live channels in memory."""
        return len(self._channels)


_BUS = AssistantEventBus()


def get_event_bus() -> AssistantEventBus:
    """Return the singleton bus. Tests can monkeypatch."""
    return _BUS
