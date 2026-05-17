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
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("event bus queue full for job=%s — dropping event", event.job_id)

    def subscribe(self) -> asyncio.Queue[AssistantEvent]:
        q: asyncio.Queue[AssistantEvent] = asyncio.Queue(maxsize=500)
        # Replay buffered history so late subscribers don't miss anything.
        for ev in self.history:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                break
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass

    def close(self) -> None:
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
    """Process-wide bus: one channel per job_id, lazy-created on first use."""

    def __init__(self) -> None:
        self._channels: dict[str, _JobChannel] = {}

    def publish(self, event: AssistantEvent) -> None:
        ch = self._channels.get(event.job_id)
        if ch is None:
            ch = _JobChannel()
            self._channels[event.job_id] = ch
        ch.publish(event)

    def subscribe(self, job_id: str) -> asyncio.Queue[AssistantEvent]:
        ch = self._channels.get(job_id)
        if ch is None:
            ch = _JobChannel()
            self._channels[job_id] = ch
        return ch.subscribe()

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


_BUS = AssistantEventBus()


def get_event_bus() -> AssistantEventBus:
    """Return the singleton bus. Tests can monkeypatch."""
    return _BUS
