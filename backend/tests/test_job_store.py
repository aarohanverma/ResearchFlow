"""Tests for the JobStore abstraction and its in-memory implementation.

The Redis adapter is exercised only in integration; here we lock down the
public contract via the in-memory adapter (which all workflows hit by default).
"""

from __future__ import annotations

import pytest

from app.services.job_store import InMemoryJobStore, get_job_store, reset_job_store_for_tests


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global JobStore singleton between tests."""
    reset_job_store_for_tests()
    yield
    reset_job_store_for_tests()


@pytest.mark.asyncio
async def test_put_then_get_roundtrip():
    store = InMemoryJobStore()
    payload = {"job_id": "j1", "user_id": "u1", "status": "queued", "created_at": "2026-01-01T00:00:00+00:00"}
    await store.put("j1", payload)
    got = await store.get("j1")
    assert got == payload


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    store = InMemoryJobStore()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_update_merges_into_existing_record():
    store = InMemoryJobStore()
    await store.put("j1", {"job_id": "j1", "user_id": "u1", "status": "queued"})
    await store.update("j1", {"status": "running", "progress": 0.5})
    got = await store.get("j1")
    assert got["status"] == "running"
    assert got["progress"] == 0.5
    assert got["user_id"] == "u1"


@pytest.mark.asyncio
async def test_update_missing_is_noop():
    store = InMemoryJobStore()
    await store.update("ghost", {"status": "running"})
    assert await store.get("ghost") is None


@pytest.mark.asyncio
async def test_list_by_user_orders_newest_first():
    store = InMemoryJobStore()
    await store.put("j1", {"user_id": "u1", "created_at": "2026-01-01T00:00:00+00:00"})
    await store.put("j2", {"user_id": "u1", "created_at": "2026-01-02T00:00:00+00:00"})
    await store.put("j3", {"user_id": "u2", "created_at": "2026-01-03T00:00:00+00:00"})

    result = await store.list_by_user("u1")
    assert len(result) == 2
    assert result[0]["created_at"] >= result[1]["created_at"]


@pytest.mark.asyncio
async def test_list_by_user_empty():
    store = InMemoryJobStore()
    assert await store.list_by_user("nobody") == []


@pytest.mark.asyncio
async def test_delete_is_idempotent():
    store = InMemoryJobStore()
    await store.put("j1", {"user_id": "u1"})
    await store.delete("j1")
    assert await store.get("j1") is None
    # Second delete must not raise
    await store.delete("j1")


@pytest.mark.asyncio
async def test_get_factory_returns_singleton():
    a = get_job_store()
    b = get_job_store()
    assert a is b


@pytest.mark.asyncio
async def test_factory_returns_in_memory_by_default(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "cache_backend", "local")
    reset_job_store_for_tests()
    store = get_job_store()
    assert isinstance(store, InMemoryJobStore)


@pytest.mark.asyncio
async def test_in_memory_isolates_payload_copies():
    """Mutating the returned payload must not affect the stored record."""
    store = InMemoryJobStore()
    await store.put("j1", {"user_id": "u1", "status": "queued"})
    got = await store.get("j1")
    assert got is not None
    got["status"] = "tampered"

    refetched = await store.get("j1")
    assert refetched["status"] == "queued"
