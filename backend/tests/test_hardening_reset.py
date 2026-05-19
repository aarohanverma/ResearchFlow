"""Regression tests for the hard-reset hardening pass.

These tests exercise the in-process pieces of ``/api/v1/dev/reset`` without
needing a live database — every layer that's safe to test in isolation:

* ``_cancel_inflight_tasks`` actually cancels and awaits known task pools
  with a bounded grace window, and never raises when a pool is missing.
* The cancellation pass is idempotent — running it twice in a row is safe.
* A task that ignores ``CancelledError`` for longer than the grace window
  does not block the reset (the helper returns after ``grace_seconds``).
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_cancel_inflight_tasks_cancels_known_pools(monkeypatch):
    """Pre-populate two known pools with sleeping tasks; reset cancels both."""
    from app.api.v1 import search as _search_mod
    from app.api.v1 import genie as _genie_mod
    from app.api.v1.dev import _cancel_inflight_tasks

    async def _slow_task() -> None:
        await asyncio.sleep(10)

    t_search = asyncio.create_task(_slow_task(), name="rs:search:test")
    t_genie = asyncio.create_task(_slow_task(), name="rs:genie:test")

    # Monkeypatch the module-level pools so we don't accidentally cancel
    # tasks unrelated to this test.
    monkeypatch.setattr(_search_mod, "_background_tasks", {t_search}, raising=False)
    monkeypatch.setattr(_genie_mod, "_background_tasks", {t_genie}, raising=False)

    await _cancel_inflight_tasks(grace_seconds=1.0)

    assert t_search.cancelled() or t_search.done()
    assert t_genie.cancelled() or t_genie.done()

    # Idempotent — calling again with empty pools must not raise.
    await _cancel_inflight_tasks(grace_seconds=0.1)


@pytest.mark.asyncio
async def test_cancel_inflight_tolerates_missing_modules(monkeypatch):
    """Each pool is best-effort — a missing attribute must not blow up the helper."""
    from app.api.v1 import search as _search_mod
    from app.api.v1.dev import _cancel_inflight_tasks

    # Remove a real module's pool attribute to simulate a stripped build.
    if hasattr(_search_mod, "_background_tasks"):
        monkeypatch.delattr(_search_mod, "_background_tasks", raising=False)
    await _cancel_inflight_tasks(grace_seconds=0.1)  # must not raise


@pytest.mark.asyncio
async def test_cancel_inflight_respects_grace_window(monkeypatch):
    """A task that swallows CancelledError must not extend the grace window."""
    from app.api.v1 import search as _search_mod
    from app.api.v1.dev import _cancel_inflight_tasks

    async def _stubborn() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # Pretend the task does cleanup that ignores the cancel.
            await asyncio.sleep(5)

    t = asyncio.create_task(_stubborn(), name="rs:stubborn:test")
    monkeypatch.setattr(_search_mod, "_background_tasks", {t}, raising=False)

    loop = asyncio.get_event_loop()
    started = loop.time()
    await _cancel_inflight_tasks(grace_seconds=0.5)
    elapsed = loop.time() - started

    # Bounded by the grace + small scheduler slack, never the full 5s.
    assert elapsed < 1.0, f"reset blocked for {elapsed:.2f}s — grace window not enforced"
    # Clean up so the test process doesn't leak the stubborn task.
    t.cancel()
    try:
        await asyncio.wait_for(t, timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
