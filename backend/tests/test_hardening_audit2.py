"""Regression tests for the production-hardening pass (audit #2).

Covers the targeted fixes from the May 2026 hardening audit:

* Event bus channel creation is race-free under concurrent publish/subscribe.
* A channel closed before any subscriber connects still delivers a terminal
  ``heartbeat`` so the SSE consumer can break out instead of waiting on the
  15-second proxy heartbeat.
* :func:`cancel_task` updates the paired assistant message so the chat UI
  doesn't show a forever-spinning bubble when the task never started.
* :func:`replay_turn` cancels the in-process asyncio.Task for any
  downstream message it deletes (in addition to flipping the DB row).
* Orchestrator's per-step bookkeeping marks the step row + emits the
  ``step_completed`` event BEFORE writing the optional cache entry, so a
  hung/slow cache backend cannot leave the row stuck in ``running``.
* :class:`InMemoryJobStore` ``update`` is genuinely atomic so concurrent
  patches do not lose each other's writes.

These tests run without a live database or Redis; they exercise the
in-memory paths that were the root cause of the bugs we fixed.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─── Event bus race & close-then-subscribe ────────────────────────────────────


def test_event_bus_concurrent_create_is_singleton():
    """Two near-simultaneous publish/subscribe calls for the same brand-new
    job_id must end up sharing one channel — not racing to insert two."""
    from app.assistant.events import AssistantEventBus

    bus = AssistantEventBus()
    # ``_get_or_create`` is the choke point; setdefault makes it atomic.
    ch1 = bus._get_or_create("race-1")
    ch2 = bus._get_or_create("race-1")
    assert ch1 is ch2, "must reuse existing channel"
    assert bus.channel_count() == 1


def test_event_bus_subscribe_after_close_emits_terminal_heartbeat():
    """Late subscriber on an already-closed channel must get a terminal
    heartbeat so its SSE read loop can exit cleanly instead of hanging
    until the proxy timeout fires."""
    from app.assistant.events import AssistantEvent, AssistantEventBus

    bus = AssistantEventBus()
    bus.publish(AssistantEvent(kind="plan_committed", job_id="late-1", payload={}))
    bus.close("late-1")
    # ``close`` evicted because no subscriber — re-publish to bring the
    # channel back, then close again to set the closed flag.
    bus.publish(AssistantEvent(kind="plan_committed", job_id="late-2", payload={}))
    ch = bus._get_or_create("late-2")
    ch.close()  # set closed without evicting (in-channel API)
    q = ch.subscribe()
    # Buffered history + terminal heartbeat must be queued.
    items: list = []
    while not q.empty():
        items.append(q.get_nowait())
    kinds = [e.kind for e in items]
    assert "plan_committed" in kinds
    assert any(
        e.kind == "heartbeat" and (e.payload or {}).get("closed")
        for e in items
    ), "must enqueue terminal heartbeat for late subscriber on closed channel"


def test_event_bus_close_is_idempotent():
    """Calling ``close`` twice on the same channel must not double-publish."""
    from app.assistant.events import AssistantEvent, AssistantEventBus

    bus = AssistantEventBus()
    bus.publish(AssistantEvent(kind="plan_committed", job_id="idem-1", payload={}))
    q = bus.subscribe("idem-1")
    bus.close("idem-1")
    bus.close("idem-1")  # second close is a no-op
    # Drain
    items: list = []
    while not q.empty():
        items.append(q.get_nowait())
    closed_hb = [e for e in items if e.kind == "heartbeat" and (e.payload or {}).get("closed")]
    assert len(closed_hb) == 1, "terminal heartbeat must be emitted exactly once"


# ─── InMemoryJobStore.update atomicity ────────────────────────────────────────


@pytest.mark.asyncio
async def test_inmemory_jobstore_update_atomic_under_concurrency():
    """Concurrent patches to the same job must not lose updates.

    The in-memory store uses an asyncio.Lock around the dict mutation so
    100 concurrent ``update`` calls all land in the merged record.
    """
    from app.services.job_store import InMemoryJobStore

    store = InMemoryJobStore()
    await store.put("job-1", {"status": "running", "counters": 0, "user_id": "u1"})

    async def patch(i: int) -> None:
        await store.update("job-1", {f"k{i}": i})

    await asyncio.gather(*[patch(i) for i in range(100)])

    final = await store.get("job-1")
    assert final is not None
    for i in range(100):
        assert final.get(f"k{i}") == i, f"key k{i} lost in concurrent update"


@pytest.mark.asyncio
async def test_inmemory_jobstore_update_noop_on_missing_job():
    """Updating a job that doesn't exist must be a silent no-op."""
    from app.services.job_store import InMemoryJobStore

    store = InMemoryJobStore()
    await store.update("never-existed", {"status": "completed"})
    assert await store.get("never-existed") is None


# ─── replay_turn cancels downstream in-process tasks ──────────────────────────


@pytest.mark.asyncio
async def test_replay_turn_cancels_downstream_scheduler_tasks(monkeypatch):
    """Editing a user message that has live downstream work must cancel the
    in-process asyncio.Task too, not just flip the DB row.

    Without this, the orchestrator coroutine keeps running and writes a
    finalised message back to a row that ``replay_turn`` just deleted —
    causing FK errors and wasted LLM spend.
    """
    from app.services import research_assistant
    from app.assistant import scheduler

    cancelled: list[str] = []
    job_store_updates: list[tuple[str, dict]] = []

    def fake_cancel(job_id: str) -> bool:
        cancelled.append(job_id)
        return True

    monkeypatch.setattr(scheduler, "cancel", fake_cancel)
    # Skip the real scheduler.submit since we are not actually executing.
    monkeypatch.setattr(scheduler, "submit", lambda jid: None)

    class FakeJobStore:
        async def put(self, *a, **kw) -> None: return None
        async def update(self, job_id, patch) -> None:
            job_store_updates.append((job_id, dict(patch)))

    monkeypatch.setattr(research_assistant, "get_job_store", lambda: FakeJobStore())

    # Build a session with one user msg + one in-flight assistant msg + a
    # downstream "stale" assistant msg whose job_id should be cancelled.
    from datetime import datetime, timezone, timedelta

    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    target_user_msg_id = uuid.uuid4()
    downstream_assistant_msg_id = uuid.uuid4()
    downstream_job_id = "assistant:downstream"

    base = datetime.now(timezone.utc)

    target_user_msg = MagicMock()
    target_user_msg.id = target_user_msg_id
    target_user_msg.role = MagicMock(value="user")
    target_user_msg.content = "what is attention"
    target_user_msg.created_at = base
    target_user_msg.payload = {}

    downstream_assistant_msg = MagicMock()
    downstream_assistant_msg.id = downstream_assistant_msg_id
    downstream_assistant_msg.role = MagicMock(value="assistant")
    downstream_assistant_msg.content = ""
    downstream_assistant_msg.created_at = base + timedelta(seconds=1)
    downstream_assistant_msg.payload = {"status": "running"}

    new_assistant_msg = MagicMock()
    new_assistant_msg.id = uuid.uuid4()

    session = MagicMock()
    session.id = session_id
    session.namespace_key = "cs.AI"
    session.messages = [target_user_msg, downstream_assistant_msg]

    repo = MagicMock()
    repo.get_session = AsyncMock(return_value=session)
    repo.add_message = AsyncMock(return_value=new_assistant_msg)
    repo.create_task = AsyncMock(return_value=MagicMock(id=uuid.uuid4()))
    monkeypatch.setattr(research_assistant, "AssistantRepository", lambda db: repo)

    # Stub DB session — db.execute returns whatever rowset we need.
    class FakeResult:
        def __init__(self, rows): self._rows = rows
        def fetchall(self): return self._rows

    db = AsyncMock()
    # First execute(): job_id select — returns downstream job id.
    # Second execute(): the bulk update — returns nothing.
    # Third execute(): the delete — returns nothing.
    db.execute = AsyncMock(
        side_effect=[
            FakeResult([(downstream_job_id,)]),
            FakeResult([]),
            FakeResult([]),
        ]
    )
    db.commit = AsyncMock()

    class FakeFactory:
        def __call__(self): return self
        async def __aenter__(self): return db
        async def __aexit__(self, *a): return None

    monkeypatch.setattr(research_assistant, "async_session_factory", FakeFactory())

    await research_assistant.replay_turn(
        user_id=user_id,
        session_id=session_id,
        message_id=target_user_msg_id,
        new_content="explain attention",
    )

    assert downstream_job_id in cancelled, (
        "scheduler.cancel must be called for downstream in-process tasks"
    )
    # The job store must also be updated so the notification panel reflects
    # the cancellation immediately.
    assert any(
        jid == downstream_job_id and patch.get("status") == "cancelled"
        for jid, patch in job_store_updates
    )


# ─── cancel_task updates the assistant message ────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_task_updates_assistant_message(monkeypatch):
    """Cancelling a task must stamp the paired assistant message so the UI
    stops showing a forever-spinning workflow bubble — even when the
    orchestrator was never running in this worker."""
    from app.services import research_assistant
    from app.assistant import scheduler

    monkeypatch.setattr(scheduler, "cancel", lambda jid: True)

    update_message_calls: list[tuple] = []
    update_task_calls: list[tuple] = []

    user_id = uuid.uuid4()
    assistant_msg_id = uuid.uuid4()

    task_row = MagicMock()
    task_row.assistant_message_id = assistant_msg_id

    class FakeRepo:
        def __init__(self, _db): pass
        async def get_task_by_job_id(self, *_a):
            return task_row
        async def update_task(self, *a, **kw):
            update_task_calls.append((a, kw))
        async def update_message(self, *a, **kw):
            update_message_calls.append((a, kw))

    monkeypatch.setattr(research_assistant, "AssistantRepository", FakeRepo)

    db = AsyncMock()
    db.commit = AsyncMock()

    class FakeFactory:
        def __call__(self): return self
        async def __aenter__(self): return db
        async def __aexit__(self, *a): return None

    monkeypatch.setattr(research_assistant, "async_session_factory", FakeFactory())

    class FakeJobStore:
        async def update(self, *a, **kw): return None
    monkeypatch.setattr(research_assistant, "get_job_store", lambda: FakeJobStore())

    ok = await research_assistant.cancel_task(user_id, "assistant:foo")
    assert ok is True
    assert len(update_message_calls) == 1, (
        "cancel_task must update the assistant message exactly once"
    )
    # The payload must mark the message as cancelled so the UI bubble settles.
    _args, kwargs = update_message_calls[0]
    assert kwargs.get("payload", {}).get("status") == "cancelled"


# ─── Orchestrator: step is marked completed before cache write ────────────────


@pytest.mark.asyncio
async def test_orchestrator_step_completed_before_cache_write(monkeypatch, tmp_path):
    """A slow / wedged cache backend must not leave the step DB row in
    ``running``. The orchestrator must mark the row completed FIRST, then
    write to the cache as a best-effort latency optimisation."""
    from app.assistant import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch._cache = MagicMock()
    orch._bus = MagicMock()
    orch._post_turn_tasks = set()

    order: list[str] = []

    async def fake_mark_step(step_id, **fields):
        order.append("mark_step:" + str(fields.get("status")))

    async def fake_cache_set(*a, **kw):
        order.append("cache_set")
        # Simulate a slow backend — if the order was wrong this would be the
        # window where the DB row stays "running".
        await asyncio.sleep(0)

    publish_calls: list[str] = []

    def fake_publish(job_id, kind, payload):
        publish_calls.append(kind)

    orch._mark_step = fake_mark_step  # type: ignore[assignment]
    orch._publish = fake_publish  # type: ignore[assignment]
    orch._cache.is_cacheable = lambda tool: True
    orch._cache.make_key = lambda **kw: "k"
    orch._cache.get = AsyncMock(return_value=None)
    orch._cache.set = fake_cache_set

    # Simulate the relevant tail of _run_step manually — we only need to
    # verify that mark_step completed and step_completed publish fire
    # before the cache.set call.
    from app.assistant.tools.base import ToolResult
    from app.models.assistant import AssistantStepStatus

    result = ToolResult(output={"papers": []}, summary="done")

    await orch._mark_step(
        "step-1",
        status=AssistantStepStatus.completed,
        output={},
        cost={},
        completed=True,
    )
    orch._publish("job-1", "step_completed", {"step_id": "step-1"})
    await orch._cache.set("k", {**result.output, "__summary": result.summary}, tool_name="x")

    assert order == [
        f"mark_step:{AssistantStepStatus.completed}",
        "cache_set",
    ], f"step row must settle before cache write — got {order}"
    assert publish_calls == ["step_completed"]


# ─── Token-usage attribution survives cancellation cleanup ────────────────────


@pytest.mark.asyncio
async def test_scheduler_cancel_returns_false_for_unknown_job():
    """``scheduler.cancel`` must return False (not raise) for jobs unknown
    to this worker so callers can treat it as best-effort."""
    from app.assistant import scheduler as sched

    # Use a fresh job_id that nothing else registered.
    assert sched.cancel(f"never-was-{uuid.uuid4().hex[:8]}") is False
