"""Regression tests for the production-hardening pass.

Targets the high-impact fixes from the audit:

* Fire-and-forget asyncio tasks are rooted in module-level sets so Python
  3.12+ cannot GC them mid-flight.
* Assistant scheduler is idempotent on double-submit.
* Event bus auto-evicts closed channels with no subscribers.
* SearchRepository.RRF surfaces semantic-only matches for manually-imported
  papers (regression for the search-after-manual-import bug).
* ``paper_import`` tool validates and dedupes arXiv IDs and stays cancellable.

These tests run without a live database — they exercise the in-memory paths
that were the root cause of the bugs we fixed.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─── Scheduler idempotency ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scheduler_double_submit_returns_same_task():
    """Two ``submit()`` calls for the same job_id while running must return
    the existing task — not start a parallel runner that would clobber DB state."""
    from app.assistant import scheduler

    started = asyncio.Event()
    block = asyncio.Event()
    call_count = 0

    async def runner(_job_id: str) -> None:
        nonlocal call_count
        call_count += 1
        started.set()
        await block.wait()

    scheduler.register_runner(runner)
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    t1 = scheduler.submit(job_id)
    await started.wait()
    t2 = scheduler.submit(job_id)

    assert t1 is t2, "double-submit must return the same Task object"
    assert call_count == 1, "runner must have been started exactly once"

    block.set()
    await asyncio.wait_for(t1, timeout=2.0)


@pytest.mark.asyncio
async def test_scheduler_submit_after_done_starts_fresh_runner():
    """After a task finishes the slot is freed and a new submit creates a new task."""
    from app.assistant import scheduler

    invocations: list[str] = []

    async def runner(job_id: str) -> None:
        invocations.append(job_id)

    scheduler.register_runner(runner)
    job_id = f"test-finish-{uuid.uuid4().hex[:8]}"
    t1 = scheduler.submit(job_id)
    await asyncio.wait_for(t1, timeout=1.0)
    t2 = scheduler.submit(job_id)
    await asyncio.wait_for(t2, timeout=1.0)

    assert t1 is not t2
    assert invocations == [job_id, job_id]


# ─── Event bus eviction ───────────────────────────────────────────────────────


def test_event_bus_close_evicts_when_no_subscribers():
    """Channels for turns that nobody subscribed to must not leak forever."""
    from app.assistant.events import AssistantEventBus, AssistantEvent

    bus = AssistantEventBus()
    bus.publish(AssistantEvent(kind="plan_committed", job_id="orphan-1", payload={}))
    assert "orphan-1" in bus._channels

    bus.close("orphan-1")  # no subscribers → immediate eviction
    assert "orphan-1" not in bus._channels


def test_event_bus_close_preserves_channel_with_subscribers():
    """A channel with live subscribers must not be evicted until they unsubscribe."""
    from app.assistant.events import AssistantEventBus, AssistantEvent

    bus = AssistantEventBus()
    bus.publish(AssistantEvent(kind="plan_committed", job_id="live-1", payload={}))
    q = bus.subscribe("live-1")
    bus.close("live-1")
    assert "live-1" in bus._channels, "must not evict while subscriber holds the queue"
    bus.unsubscribe("live-1", q)
    assert "live-1" not in bus._channels, "unsubscribe of last subscriber should evict"


# ─── Tracking adapter task rooting ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tracking_adapter_roots_recording_tasks(monkeypatch):
    """``_spawn_recording`` must add the task to the module-level set so the
    Python 3.12+ event loop does not GC it before the DB write completes."""
    from app.adapters.llm import tracking

    async def fake_record(**_kwargs):
        await asyncio.sleep(0)

    monkeypatch.setattr(tracking, "_record_usage", fake_record)

    pre_count = len(tracking._tracking_tasks)
    # _spawn_recording does not return the task, so capture by snapshot of the set.
    tracking._spawn_recording(fake_record())
    new_tasks = tracking._tracking_tasks - set()  # snapshot copy
    assert len(new_tasks) == pre_count + 1, "task must be rooted while running"

    # Drain by awaiting the new task explicitly so the done_callback runs.
    spawned = [t for t in tracking._tracking_tasks][0]
    await spawned
    # Yield once more so callbacks scheduled on done can execute.
    await asyncio.sleep(0)
    assert len(tracking._tracking_tasks) == pre_count, "done_callback must discard the task"


# ─── Search RRF: manually-imported semantic-only papers surface ───────────────


def test_rrf_keeps_manually_imported_semantic_only_match():
    """Regression: a manually-imported paper that matches ONLY semantically
    must still appear in basic search results (semantic-only auto-ingested
    papers are still dropped — that path is unchanged)."""
    from app.repositories.search import SearchRepository

    repo = SearchRepository(db=MagicMock())

    sem_results = [
        {
            "paper_id": "p-imported",
            "external_id": "1706.03762",
            "title": "Attention Is All You Need",
            "is_manually_imported": True,
            "namespace_key": "cs.LG",
        },
        {
            "paper_id": "p-auto",
            "external_id": "2401.99999",
            "title": "Generic Paper",
            "is_manually_imported": False,
            "namespace_key": "cs.LG",
        },
    ]
    kw_results: list[dict] = []  # no keyword match for either

    fused = repo._rrf_fuse(kw_results, sem_results)
    pids = {r["paper_id"] for r in fused}
    assert "p-imported" in pids, "manually-imported semantic-only match must surface"
    assert "p-auto" not in pids, "auto-ingested semantic-only match must still be dropped"


def test_rrf_promotes_keyword_match_to_hybrid():
    """Sanity check: a paper appearing in both lists must be marked hybrid."""
    from app.repositories.search import SearchRepository

    repo = SearchRepository(db=MagicMock())
    row = {
        "paper_id": "p1",
        "external_id": "ext-1",
        "title": "Both",
        "is_manually_imported": False,
        "namespace_key": "cs.AI",
    }
    fused = repo._rrf_fuse([row], [row])
    assert fused[0]["match_type"] == "hybrid"


# ─── paper_import tool ID normalisation ───────────────────────────────────────


def test_paper_import_normalises_arxiv_inputs():
    """``_normalise_arxiv_id`` strips URL prefixes and version suffixes uniformly."""
    from app.assistant.tools.paper_import import _normalise_arxiv_id

    assert _normalise_arxiv_id("1706.03762") == "1706.03762"
    assert _normalise_arxiv_id("1706.03762v3") == "1706.03762"
    assert _normalise_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert _normalise_arxiv_id("arxiv:1706.03762") == "1706.03762"
    assert _normalise_arxiv_id("arXiv:1706.03762v2") == "1706.03762"
    assert _normalise_arxiv_id("not-an-id") is None
    assert _normalise_arxiv_id("") is None


@pytest.mark.asyncio
async def test_paper_import_rejects_all_invalid_inputs_without_calling_arxiv(monkeypatch):
    """All-invalid input must short-circuit before any arXiv HTTP call."""
    from app.assistant.tools.base import ToolContext
    from app.assistant.tools.paper_import import PaperImportInput, paper_import_tool, _fetch_arxiv_entry

    calls: list[str] = []

    async def stub_fetch(_aid: str) -> dict | None:
        calls.append(_aid)
        return None

    monkeypatch.setattr("app.assistant.tools.paper_import._fetch_arxiv_entry", stub_fetch)

    async def no_cancel() -> bool:
        return False

    progress_events: list[tuple[int, str]] = []

    async def emit_progress(percent: int, msg: str) -> None:
        progress_events.append((percent, msg))

    ctx = ToolContext(
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        namespace_key="cs.AI",
        namespace_keys=["cs.AI"],
        orientation="research",
        expertise_level="practitioner",
        job_id=f"test-{uuid.uuid4().hex[:8]}",
        parent_message_id=uuid.uuid4(),
        db=MagicMock(),
        should_cancel=no_cancel,
        emit_progress=emit_progress,
    )
    params = PaperImportInput(arxiv_ids=["banana", "not-an-id"])

    result = await paper_import_tool.run(ctx, params)

    assert calls == [], "must not hit arXiv when every input is invalid"
    assert result.output["imported"] == 0
    assert result.output["failed"] == 2
    assert all(item["status"] == "invalid_id" for item in result.output["items"])


# ─── Graph clear cancels in-flight builds ─────────────────────────────────────


@pytest.mark.asyncio
async def test_clear_graph_cancels_active_build_task(monkeypatch):
    """In-flight Build Deep tasks for a cleared namespace must be cancelled
    BEFORE the DELETE statements run, preventing partial-state corruption."""
    from app.api.v1 import graph as graph_router

    # Fake cache that records get/set/delete calls
    cache_state: dict[str, dict] = {}

    class FakeCache:
        async def get(self, k):
            return cache_state.get(k)

        async def set(self, k, v, ttl_seconds=None):
            cache_state[k] = v

        async def delete(self, k):
            cache_state.pop(k, None)

    fake_cache = FakeCache()
    monkeypatch.setattr(graph_router, "get_cache", lambda: fake_cache)

    # Plant a fake running build task
    running = asyncio.Event()

    async def long_build():
        running.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(long_build(), name="build:test")
    graph_router._BUILD_TASKS["job-x"] = task
    cache_state["graph:build:job-x"] = {
        "namespace_key": "cs.AI",
        "lock_key": "graph:build:lock:cs.AI",
        "status": "running",
        "cancel_requested": False,
    }
    cache_state["graph:build:lock:cs.AI"] = "job-x"
    await running.wait()

    # Fake DB session with execute/commit
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: []))
    db.commit = AsyncMock()

    # Patch GraphService import inside the function so cache invalidations
    # don't touch real Redis / in-memory caches we don't control here.
    from app.services import graph as graph_service_mod
    monkeypatch.setattr(graph_service_mod.GraphService, "clear_subgraph_cache",
                        AsyncMock())
    graph_service_mod.GraphService._build_cache = {}

    response = await graph_router.clear_graph(
        db=db, user_id=uuid.uuid4(), namespace_keys="cs.AI",
    )

    assert "job-x" in response["cancelled_jobs"]
    assert task.cancelled() or task.done()
    # Cleanup any remaining task
    if not task.done():
        task.cancel()
