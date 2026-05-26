"""Tests for the memory-revision audit trail and the restore endpoint.

Two correctness properties matter here:

1. **Every persistent-tier write produces an audit row.** Auto-memory
   writes, manual ``memory_write`` calls, and user-initiated deletes
   from Settings all funnel through :func:`record_revision`. A
   regression that silently skips the audit row would break the
   user-facing history view without any other visible symptom.

2. **Restore is cross-user safe and recreates the live entry.** The
   restore endpoint validates revision ownership by ``user_id`` and
   resolves the bucket via the root session. A request to restore
   another user's revision must 404 — not 403, so we don't leak
   existence information.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.settings import (
    MemoryRestoreRequest,
    restore_memory_revision,
)
from app.assistant.memory_revisions import (
    derive_entry_status,
    record_revision,
    split_subject_topic,
)


@asynccontextmanager
async def _noop_lock(_sid):
    yield


# ── split_subject_topic ──────────────────────────────────────────────────────


def test_split_subject_topic_canonical():
    assert split_subject_topic("cs.AI") == ("cs", "AI")
    assert split_subject_topic("physics.quant-ph") == ("physics", "quant-ph")


def test_split_subject_topic_no_separator():
    """A namespace without a dot is treated as the subject; topic empty."""
    assert split_subject_topic("standalone_ns") == ("standalone_ns", "")
    assert split_subject_topic("") == ("", "")


def test_split_subject_topic_trims_oversize_segments():
    long_subject = "x" * 100
    s, t = split_subject_topic(f"{long_subject}.topic")
    assert len(s) <= 60
    assert t == "topic"


# ── derive_entry_status ──────────────────────────────────────────────────────


def test_derive_entry_status_active_when_no_ttl():
    """No TTL → never stale; entries are evergreen by default."""
    entry = {"value": "x", "type": "finding", "ts": "2026-05-25T10:00:00+00:00"}
    assert derive_entry_status(entry) == "active"


def test_derive_entry_status_stale_past_ttl():
    """An entry past its TTL window must be marked stale so the UI
    can flag it and the user can decide to refresh / delete."""
    # 30 days ago with a 7-day ttl → stale.
    past_ts = "2025-01-01T00:00:00+00:00"
    entry = {"value": "x", "type": "finding", "ts": past_ts, "ttl_days": 7}
    # Pass a now_iso well after past_ts so the staleness check fires
    # deterministically regardless of wall-clock.
    assert derive_entry_status(entry, now_iso="2026-05-25T10:00:00+00:00") == "stale"


def test_derive_entry_status_legacy_string_entry():
    """Pre-versioning entries stored as bare strings must still
    surface as ``active`` without crashing the projection."""
    assert derive_entry_status("legacy raw string") == "active"


# ── record_revision: safety guards ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_revision_refuses_unknown_action():
    """Unknown actions silently no-op (best-effort audit trail) —
    the function must never raise even when called incorrectly."""
    db = AsyncMock()
    ok = await record_revision(
        db,
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        tier="long",
        key="x",
        value="y",
        action="not_a_real_action",
    )
    assert ok is False


@pytest.mark.asyncio
async def test_record_revision_refuses_empty_key():
    db = AsyncMock()
    ok = await record_revision(
        db,
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        tier="long",
        key="",
        value="y",
        action="create",
    )
    assert ok is False


@pytest.mark.asyncio
async def test_record_revision_refuses_unknown_tier():
    db = AsyncMock()
    ok = await record_revision(
        db,
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        tier="exotic_tier",
        key="x",
        value="y",
        action="create",
    )
    assert ok is False


@pytest.mark.asyncio
async def test_record_revision_swallows_db_failure():
    """A DB error during the audit write must NEVER propagate — the
    live memory write must be allowed to commit even if auditing
    fails. The function returns False on any internal exception."""
    db = MagicMock()
    db.add = MagicMock(side_effect=RuntimeError("simulated DB failure"))
    # db.get returns a mock root session via async path
    root = MagicMock()
    root.id = uuid.uuid4()
    root.parent_session_id = None
    db.get = AsyncMock(return_value=root)

    ok = await record_revision(
        db,
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        tier="long",
        key="x",
        value="y",
        action="create",
        namespace_key="cs.AI",
    )
    assert ok is False


# ── Restore endpoint: cross-user safety ──────────────────────────────────────


@pytest.mark.asyncio
async def test_restore_revision_404s_for_other_users_revision():
    """A user must never be able to restore another user's revision
    even if they guess the revision UUID. The endpoint returns 404
    (not 403) so existence isn't leaked."""
    from fastapi import HTTPException

    me = uuid.uuid4()
    db = AsyncMock()
    # Ownership query returns nothing because the user_id filter
    # eliminates the other user's row.
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)

    with pytest.raises(HTTPException) as exc:
        await restore_memory_revision(
            body=MemoryRestoreRequest(revision_id=uuid.uuid4()),
            user_id=me,
            db=db,
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_restore_revision_rejects_empty_value():
    """A pure delete revision (value='', previous_value='') has
    nothing to restore — must 400 rather than silently restore an
    empty string into the live entry."""
    from fastapi import HTTPException

    me = uuid.uuid4()
    revision = MagicMock()
    revision.id = uuid.uuid4()
    revision.user_id = me
    revision.root_session_id = uuid.uuid4()
    revision.tier = "long"
    revision.namespace_key = "cs.AI"
    revision.key = "ghost"
    revision.value = ""
    revision.previous_value = None
    revision.entry_type = "context"

    root = MagicMock()
    root.id = revision.root_session_id
    root.user_id = me
    root.state = {}

    db = AsyncMock()
    seq = [revision, root]  # first call returns revision; second returns root
    def _exec_side_effect(_stmt):
        r = MagicMock()
        r.scalar_one_or_none.return_value = seq.pop(0) if seq else None
        return r
    db.execute = AsyncMock(side_effect=_exec_side_effect)

    with patch("app.assistant.state_lock.session_state_lock", _noop_lock):
        with pytest.raises(HTTPException) as exc:
            await restore_memory_revision(
                body=MemoryRestoreRequest(revision_id=revision.id),
                user_id=me,
                db=db,
            )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_restore_revision_recreates_long_tier_entry():
    """Restoring a prior revision reapplies its value to the live
    namespace bucket. The entry comes back at the same key with the
    revision's value; ``source`` becomes ``"restore"`` so the
    provenance trail is honest."""
    me = uuid.uuid4()
    rev_id = uuid.uuid4()
    root_id = uuid.uuid4()

    revision = MagicMock()
    revision.id = rev_id
    revision.user_id = me
    revision.root_session_id = root_id
    revision.tier = "long"
    revision.namespace_key = "cs.AI"
    revision.key = "user_pref"
    revision.value = "User prefers concise technical answers."
    revision.previous_value = None
    revision.entry_type = "preference"

    root = MagicMock()
    root.id = root_id
    root.user_id = me
    root.state = {"ns_memory": {"cs.AI": {}}}

    db = AsyncMock()
    seq = [revision, root]
    def _exec_side_effect(_stmt):
        r = MagicMock()
        r.scalar_one_or_none.return_value = seq.pop(0) if seq else None
        return r
    db.execute = AsyncMock(side_effect=_exec_side_effect)
    db.commit = AsyncMock()
    # record_revision will call db.get to resolve root — return our root
    db.get = AsyncMock(return_value=root)
    db.add = MagicMock()

    with patch("app.assistant.state_lock.session_state_lock", _noop_lock):
        out = await restore_memory_revision(
            body=MemoryRestoreRequest(revision_id=rev_id),
            user_id=me,
            db=db,
        )
    assert out["restored"] is True
    assert out["key"] == "user_pref"
    # Live state now carries the restored value.
    restored = root.state["ns_memory"]["cs.AI"]["user_pref"]
    assert restored["value"] == "User prefers concise technical answers."
    assert restored["source"] == "restore"
    assert restored["restored_from_revision"] == str(rev_id)
