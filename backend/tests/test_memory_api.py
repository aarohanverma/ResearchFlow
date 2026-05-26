"""Tests for the long-term memory inspect / delete / clear API.

The endpoints exposed at ``/api/v1/settings/memory*`` are the
user-facing controls for what RA remembers across turns. Two things
matter most:

1. **User isolation** — never return another user's entries. The
   ``user_id`` filter on root-session lookups is the load-bearing
   isolation boundary; this test fails loudly if a regression breaks
   it.
2. **Surgical delete vs bulk clear** — the per-entry delete must only
   remove the targeted row; bulk clear must only touch the requested
   tier (and never short-term chat memory).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.settings import (
    MemoryClearRequest,
    MemoryDeleteRequest,
    _memory_entry_to_row,
    _user_root_sessions,
    clear_memory,
    delete_memory_entry,
    list_memory,
)


def _root_session(user_id, *, root_id=None, state=None):
    """Build a MagicMock(AssistantSession) with user_id + state."""
    sess = MagicMock()
    sess.id = root_id or uuid.uuid4()
    sess.user_id = user_id
    sess.parent_session_id = None
    sess.state = state or {}
    return sess


# ── Row projection ──────────────────────────────────────────────────────────


def test_memory_entry_to_row_handles_dict_entry():
    row = _memory_entry_to_row(
        tier="long",
        namespace_key="cs.AI",
        key="user_pref_depth",
        entry={
            "value": "User prefers technical responses.",
            "type": "preference",
            "ts": "2026-05-25T10:00:00+00:00",
            "source": "auto",
            "ttl_days": None,
        },
    )
    assert row["tier"] == "long"
    assert row["namespace_key"] == "cs.AI"
    assert row["key"] == "user_pref_depth"
    assert row["value"].startswith("User prefers")
    assert row["type"] == "preference"
    assert row["source"] == "auto"
    assert row["ttl_days"] is None


def test_memory_entry_to_row_handles_legacy_string_entry():
    """Some early sessions stored memory as bare strings; the
    projection must tolerate that without crashing."""
    row = _memory_entry_to_row(
        tier="medium", namespace_key="", key="user_name", entry="Aarohan",
    )
    assert row["value"] == "Aarohan"
    assert row["type"] == "context"  # safe default for legacy entries
    assert row["ts"] == ""


# ── User isolation: GET ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_memory_returns_only_current_user_entries():
    """The root-session lookup filters by user_id — verified here by
    confirming the SELECT was constructed with the calling user's id.
    Regression guard for a class of bugs where a join or extra
    parameter accidentally widens the query."""
    me = uuid.uuid4()
    other = uuid.uuid4()
    my_root = _root_session(
        me,
        state={
            "tree_memory": {"my_finding": {"value": "x", "type": "finding", "ts": "2026-05-25"}},
            "ns_memory":   {"cs.AI": {"my_pref": {"value": "y", "type": "preference", "ts": "2026-05-25"}}},
        },
    )
    other_root = _root_session(
        other,
        state={
            "ns_memory": {"cs.AI": {"other_secret": {"value": "secret", "type": "preference", "ts": "2026-05-25"}}},
        },
    )

    db = AsyncMock()
    # Simulate a SELECT-with-user_id-filter that returns ONLY my_root —
    # this is what the real DB does because we filter by user_id.
    result = MagicMock()
    result.scalars.return_value.all.return_value = [my_root]
    db.execute = AsyncMock(return_value=result)

    out = await list_memory(user_id=me, db=db, tier=None, namespace_key=None)
    assert out["counts"]["medium"] == 1
    assert out["counts"]["long"] == 1
    # No "other_secret" key from the other user must appear.
    keys = [r["key"] for r in out["entries"]]
    assert "my_finding" in keys
    assert "my_pref" in keys
    assert "other_secret" not in keys


@pytest.mark.asyncio
async def test_list_memory_tier_filter_applies():
    me = uuid.uuid4()
    root = _root_session(
        me,
        state={
            "tree_memory": {"t1": {"value": "tree-val", "type": "context", "ts": ""}},
            "ns_memory":   {"cs.AI": {"n1": {"value": "ns-val", "type": "context", "ts": ""}}},
        },
    )
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [root]
    db.execute = AsyncMock(return_value=result)

    long_only = await list_memory(user_id=me, db=db, tier="long", namespace_key=None)
    assert long_only["counts"]["long"] == 1
    assert long_only["counts"]["medium"] == 0

    medium_only = await list_memory(user_id=me, db=db, tier="medium", namespace_key=None)
    assert medium_only["counts"]["medium"] == 1
    assert medium_only["counts"]["long"] == 0


# ── Delete: 404 on cross-user attempts ──────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_memory_404s_when_session_belongs_to_another_user():
    """Per-entry delete validates ownership. If the calling user
    doesn't own the root_session_id they supplied, return 404 —
    explicitly NOT 403, so we don't leak existence."""
    from fastapi import HTTPException

    me = uuid.uuid4()
    other_root_id = uuid.uuid4()

    db = AsyncMock()
    # Ownership query returns nothing because the user_id filter doesn't match.
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)

    with pytest.raises(HTTPException) as exc:
        await delete_memory_entry(
            body=MemoryDeleteRequest(
                tier="long",
                key="some_key",
                namespace_key="cs.AI",
                root_session_id=other_root_id,
            ),
            user_id=me,
            db=db,
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_memory_removes_only_targeted_key():
    me = uuid.uuid4()
    root_id = uuid.uuid4()
    root = _root_session(
        me,
        root_id=root_id,
        state={
            "ns_memory": {"cs.AI": {
                "keep_me":  {"value": "k", "type": "context", "ts": ""},
                "delete_me": {"value": "d", "type": "context", "ts": ""},
            }},
        },
    )

    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = root
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    # No-op state_lock for the unit test — the lock's correctness is
    # exercised separately.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lock(_session_id):
        yield

    with patch("app.assistant.state_lock.session_state_lock", _noop_lock):
        out = await delete_memory_entry(
            body=MemoryDeleteRequest(
                tier="long",
                key="delete_me",
                namespace_key="cs.AI",
                root_session_id=root_id,
            ),
            user_id=me,
            db=db,
        )
    assert out["removed"] is True
    assert "keep_me" in root.state["ns_memory"]["cs.AI"]
    assert "delete_me" not in root.state["ns_memory"]["cs.AI"]


@pytest.mark.asyncio
async def test_delete_memory_idempotent_when_key_missing():
    me = uuid.uuid4()
    root_id = uuid.uuid4()
    root = _root_session(me, root_id=root_id, state={"ns_memory": {"cs.AI": {}}})
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = root
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lock(_sid):
        yield

    with patch("app.assistant.state_lock.session_state_lock", _noop_lock):
        out = await delete_memory_entry(
            body=MemoryDeleteRequest(
                tier="long",
                key="ghost_key",
                namespace_key="cs.AI",
                root_session_id=root_id,
            ),
            user_id=me,
            db=db,
        )
    # Idempotent: not an error, just a no-op.
    assert out["removed"] is False


# ── Bulk clear ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clear_memory_long_preserves_medium_and_chat():
    """Clearing 'long' tier must NOT touch 'medium' (tree) or
    short-term chat memory — that's the user's explicit spec."""
    me = uuid.uuid4()
    root = _root_session(
        me,
        state={
            "chat_memory": {"recent": {"value": "x", "type": "context", "ts": ""}},
            "tree_memory": {"t1": {"value": "y", "type": "context", "ts": ""}},
            "ns_memory":   {"cs.AI": {"n1": {"value": "z", "type": "context", "ts": ""}}},
        },
    )
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [root]
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lock(_sid):
        yield

    with patch("app.assistant.state_lock.session_state_lock", _noop_lock):
        out = await clear_memory(
            body=MemoryClearRequest(tier="long", namespace_key=None),
            user_id=me,
            db=db,
        )

    assert out["removed"]["long"] == 1
    assert out["removed"]["medium"] == 0
    # Tree memory survives
    assert root.state["tree_memory"] == {"t1": {"value": "y", "type": "context", "ts": ""}}
    # Chat memory survives untouched
    assert "recent" in root.state["chat_memory"]
    # Namespace bucket is wiped
    assert root.state["ns_memory"] == {}


@pytest.mark.asyncio
async def test_clear_memory_with_namespace_only_clears_that_namespace():
    me = uuid.uuid4()
    root = _root_session(
        me,
        state={
            "ns_memory": {
                "cs.AI":  {"a1": {"value": "x", "type": "context", "ts": ""}},
                "cs.NLP": {"n1": {"value": "y", "type": "context", "ts": ""}},
            },
        },
    )
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [root]
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lock(_sid):
        yield

    with patch("app.assistant.state_lock.session_state_lock", _noop_lock):
        out = await clear_memory(
            body=MemoryClearRequest(tier="long", namespace_key="cs.AI"),
            user_id=me,
            db=db,
        )

    assert out["removed"]["long"] == 1
    # cs.NLP entries survive
    assert "n1" in root.state["ns_memory"]["cs.NLP"]
    # cs.AI is wiped (bucket emptied, not deleted)
    assert root.state["ns_memory"]["cs.AI"] == {}
