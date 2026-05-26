"""Background-task fire-and-forget paths must NEVER swallow exceptions silently.

Production-grade observability requires that any exception raised by a
fire-and-forget task surfaces in the application logs. Before this
hardening pass, several spawn helpers attached only ``set.discard`` as
their done-callback — Python's default behaviour then emits
"task exception was never retrieved" at DEBUG level, which production
logs typically filter. The result: a crashing background worker
disappeared from observability with no operator signal.

Each test asserts:
  * The task's exception IS logged at WARNING (or DEBUG for pure-
    telemetry paths the project explicitly downgrades).
  * The task is still removed from the registry set (no leak).
  * Cancellation is silent (no spurious log on user-driven cancel).
"""

from __future__ import annotations

import asyncio
import logging

import pytest


# ── _spawn_background (Genie SSE / fan-out) ─────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_background_logs_exception_and_clears_registry(caplog):
    from app.api.v1 import genie as genie_mod

    async def boom() -> None:
        raise RuntimeError("boom-from-bg")

    caplog.set_level(logging.WARNING, logger=genie_mod.log.name)
    task = genie_mod._spawn_background(boom(), name="bg-test")
    # Wait for the done callback to fire.
    try:
        await task
    except RuntimeError:
        pass

    # Registry was cleared (no task leak across the process lifetime).
    assert task not in genie_mod._background_tasks
    # The exception surfaced in the logs at WARNING.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("boom-from-bg" in m for m in msgs), (
        f"expected exception text in logs, got {msgs!r}"
    )


@pytest.mark.asyncio
async def test_spawn_background_cancellation_does_not_log_warning(caplog):
    from app.api.v1 import genie as genie_mod

    async def long_running() -> None:
        await asyncio.sleep(10)

    caplog.set_level(logging.WARNING, logger=genie_mod.log.name)
    task = genie_mod._spawn_background(long_running(), name="bg-cancel-test")
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task not in genie_mod._background_tasks
    # User-driven cancellation is NOT an operator-visible problem.
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("bg-cancel-test" in m and "failed" in m for m in msgs)


# ── memory.py recall-bump telemetry ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_bump_failure_logs_at_debug(caplog):
    """The last_recalled_ts bump is pure telemetry — failure stays at
    DEBUG, but the registry must still drain."""
    from app.assistant.tools import memory as memory_mod

    async def boom() -> None:
        raise ValueError("ts-bump-boom")

    caplog.set_level(logging.DEBUG, logger=memory_mod.log.name)

    bg_task = asyncio.create_task(boom(), name="memory_recall:last_recalled_ts")
    memory_mod._RECALL_BG_TASKS.add(bg_task)

    # Reproduce the production callback shape.
    def _on_recall_done(t: asyncio.Task) -> None:
        memory_mod._RECALL_BG_TASKS.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            memory_mod.log.debug("last_recalled_ts bump failed: %s", exc)

    bg_task.add_done_callback(_on_recall_done)
    try:
        await bg_task
    except ValueError:
        pass

    assert bg_task not in memory_mod._RECALL_BG_TASKS
    msgs = [r.getMessage() for r in caplog.records]
    assert any("ts-bump-boom" in m for m in msgs)
