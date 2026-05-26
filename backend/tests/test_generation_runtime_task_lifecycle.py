"""Regression tests for the generation-runtime background task lifecycle.

The ``queue_generation_job`` helper spawns a background asyncio.Task
that runs the actual workflow. Two properties this layer must hold:

1. ``_ACTIVE_TASKS`` must NEVER leak — every task that's registered
   must be removed once it completes, regardless of whether it ran to
   completion, raised, or was cancelled before its inner finally
   could fire (rare at shutdown but possible).

2. An uncaught exception inside the workflow must be SURFACED in the
   logs — not disappear as a "task exception was never retrieved"
   debug warning that production log filters drop. Without this, a
   broken workflow could silently fail with the artifact stuck in
   ``running`` state and no operator signal.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

import pytest

from app.workflows import _generation_runtime as gr


@pytest.mark.asyncio
async def test_queue_generation_job_clears_active_tasks_on_success(monkeypatch):
    """A successful runner must leave ``_ACTIVE_TASKS`` empty."""
    # Avoid touching the real JobStore — its put() is best-effort, but
    # we still want a fast pure-asyncio test.
    fake_store = type("FakeStore", (), {"put": lambda self, *a, **kw: _noop()})()
    monkeypatch.setattr(gr, "get_job_store", lambda: fake_store)

    artifact_id = uuid4()

    async def runner(_job_id: str) -> None:
        await asyncio.sleep(0)

    job_id = gr.queue_generation_job(
        artifact_id=artifact_id,
        user_id=uuid4(),
        source_type="paper",
        source_id="src",
        expertise_level="practitioner",
        orientation="both",
        generation_type="podcast",
        title="t",
        runner=runner,
    )
    assert isinstance(job_id, str)

    # Wait for the bootstrap task to finish.
    for _ in range(50):
        if str(artifact_id) not in gr._ACTIVE_TASKS:
            break
        await asyncio.sleep(0.01)

    assert str(artifact_id) not in gr._ACTIVE_TASKS, (
        "_ACTIVE_TASKS must be drained after success"
    )


@pytest.mark.asyncio
async def test_queue_generation_job_logs_uncaught_exception(monkeypatch, caplog):
    """A runner that raises an uncaught exception must produce a WARNING
    log line — the prior gap was that exceptions disappeared as
    asyncio debug warnings filtered out in production."""
    fake_store = type("FakeStore", (), {"put": lambda self, *a, **kw: _noop()})()
    monkeypatch.setattr(gr, "get_job_store", lambda: fake_store)

    artifact_id = uuid4()

    async def boom_runner(_job_id: str) -> None:
        raise RuntimeError("simulated workflow boom")

    caplog.set_level(logging.WARNING, logger=gr.log.name)
    gr.queue_generation_job(
        artifact_id=artifact_id,
        user_id=uuid4(),
        source_type="paper",
        source_id="src",
        expertise_level="practitioner",
        orientation="both",
        generation_type="slides",
        title="t",
        runner=boom_runner,
    )

    for _ in range(100):
        if str(artifact_id) not in gr._ACTIVE_TASKS:
            break
        await asyncio.sleep(0.01)

    assert str(artifact_id) not in gr._ACTIVE_TASKS
    msgs = [r.getMessage() for r in caplog.records]
    assert any("simulated workflow boom" in m for m in msgs), (
        f"expected exception to be logged at WARNING, got {msgs!r}"
    )


@pytest.mark.asyncio
async def test_queue_generation_job_cancellation_does_not_log_warning(
    monkeypatch, caplog,
):
    """User-driven cancellation is NOT an operator-visible problem and
    must NOT produce a WARNING log line. (Otherwise every user-cancelled
    podcast would pollute the warning channel.)"""
    fake_store = type("FakeStore", (), {"put": lambda self, *a, **kw: _noop()})()
    monkeypatch.setattr(gr, "get_job_store", lambda: fake_store)

    artifact_id = uuid4()

    async def long_runner(_job_id: str) -> None:
        await asyncio.sleep(10)

    caplog.set_level(logging.WARNING, logger=gr.log.name)
    gr.queue_generation_job(
        artifact_id=artifact_id,
        user_id=uuid4(),
        source_type="paper",
        source_id="src",
        expertise_level="practitioner",
        orientation="both",
        generation_type="podcast",
        title="t",
        runner=long_runner,
    )

    # Cancel the underlying task and wait for it to drain.
    task = gr._ACTIVE_TASKS.get(str(artifact_id))
    assert task is not None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert str(artifact_id) not in gr._ACTIVE_TASKS
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any(
        "failed" in m and str(artifact_id) in m for m in warning_msgs
    ), f"cancellation must not log a failure warning, got {warning_msgs!r}"


async def _noop() -> None:
    return None
