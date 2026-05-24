"""Tests for the in-process HITL inbox used by the ReAct loop's
``hitl_gate`` middleware. The inbox is process-local but otherwise
behaves like a small async pub/sub: ``register_pending`` returns a
future, ``resolve`` fulfils it, ``peek`` is a non-blocking read.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.assistant import hitl_inbox


@pytest.fixture(autouse=True)
def _clear_inbox():
    """Each test starts with an empty registry — the module-level dict
    is global, so test isolation requires an explicit reset."""
    hitl_inbox._INBOX.clear()  # noqa: SLF001 — test fixture
    yield
    hitl_inbox._INBOX.clear()  # noqa: SLF001


@pytest.mark.asyncio
async def test_register_and_resolve_roundtrip():
    sid = str(uuid.uuid4())
    uid = str(uuid.uuid4())
    rec = await hitl_inbox.register_pending(
        session_id=sid, user_id=uid, tool="genie_synthesize",
        params={"paper_ids": ["a", "b"]}, preview={"summary": "test"},
    )
    assert hitl_inbox.peek(rec.request_id) is rec

    ok = hitl_inbox.resolve(
        request_id=rec.request_id, session_id=sid, user_id=uid,
        decision=hitl_inbox.HitlDecision(status="approve"),
    )
    assert ok is True
    decision = await asyncio.wait_for(rec.future, timeout=1.0)
    assert decision.status == "approve"
    # Slot evicted on resolve.
    assert hitl_inbox.peek(rec.request_id) is None


@pytest.mark.asyncio
async def test_resolve_ownership_mismatch_refused():
    sid = str(uuid.uuid4())
    uid = str(uuid.uuid4())
    rec = await hitl_inbox.register_pending(
        session_id=sid, user_id=uid, tool="genie_synthesize",
        params={}, preview={},
    )
    # Wrong session id.
    ok = hitl_inbox.resolve(
        request_id=rec.request_id, session_id=str(uuid.uuid4()),
        user_id=uid,
        decision=hitl_inbox.HitlDecision(status="approve"),
    )
    assert ok is False
    # Wrong user id.
    ok = hitl_inbox.resolve(
        request_id=rec.request_id, session_id=sid,
        user_id=str(uuid.uuid4()),
        decision=hitl_inbox.HitlDecision(status="approve"),
    )
    assert ok is False
    # Slot remains pending.
    assert hitl_inbox.peek(rec.request_id) is rec


@pytest.mark.asyncio
async def test_resolve_unknown_id_returns_false():
    ok = hitl_inbox.resolve(
        request_id="does-not-exist", session_id="x", user_id="y",
        decision=hitl_inbox.HitlDecision(status="approve"),
    )
    assert ok is False


@pytest.mark.asyncio
async def test_discard_removes_slot():
    rec = await hitl_inbox.register_pending(
        session_id="s", user_id="u", tool="t", params={}, preview={},
    )
    hitl_inbox.discard(rec.request_id)
    assert hitl_inbox.peek(rec.request_id) is None
    # Double-discard must not raise.
    hitl_inbox.discard(rec.request_id)


@pytest.mark.asyncio
async def test_modify_decision_carries_params():
    sid, uid = "s", "u"
    rec = await hitl_inbox.register_pending(
        session_id=sid, user_id=uid, tool="genie_synthesize",
        params={"paper_ids": ["a"]}, preview={},
    )
    hitl_inbox.resolve(
        request_id=rec.request_id, session_id=sid, user_id=uid,
        decision=hitl_inbox.HitlDecision(
            status="modify", params={"paper_ids": ["a", "b", "c"]},
        ),
    )
    decision = await rec.future
    assert decision.status == "modify"
    assert decision.params == {"paper_ids": ["a", "b", "c"]}


@pytest.mark.asyncio
async def test_register_binds_future_to_running_loop():
    """The pending future must be bound to the currently-running event
    loop so the awaiting middleware actually observes ``set_result``
    rather than blocking against a stale loop's future."""
    rec = await hitl_inbox.register_pending(
        session_id="s", user_id="u", tool="t", params={}, preview={},
    )
    assert rec.future.get_loop() is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_hard_cap_evicts_oldest_with_timeout_signal():
    """The hard-cap eviction path must resolve the evicted slot's
    future with a timeout decision so a blocked middleware doesn't
    deadlock on a slot the registry has dropped underneath it."""
    # Stand up exactly _INBOX_HARD_CAP records, then add one more.
    cap = hitl_inbox._INBOX_HARD_CAP  # noqa: SLF001 — test inspection
    records = []
    for _ in range(cap):
        records.append(await hitl_inbox.register_pending(
            session_id="s", user_id="u", tool="t", params={}, preview={},
        ))
    evicted = records[0]
    overflow = await hitl_inbox.register_pending(
        session_id="s", user_id="u", tool="t", params={}, preview={},
    )
    # Oldest got evicted with a timeout decision.
    assert evicted.future.done()
    decision = evicted.future.result()
    assert decision.status == "timeout"
    # Newest is registered cleanly.
    assert hitl_inbox.peek(overflow.request_id) is overflow
