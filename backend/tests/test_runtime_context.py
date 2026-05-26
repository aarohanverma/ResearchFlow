"""Tests for the typed RuntimeContext wrapper.

The wrapper exposes ToolContext fields as typed properties plus adds
permission scopes derived from the User row. Two correctness
properties matter:

  1. ``is_allowed`` and ``require`` enforce permissions deterministically
     (admin wildcard, fine-grained scopes, missing scopes).
  2. ``build_runtime_for_user`` reads the User row safely — admin → ``*``,
     feature_overrides → fine-grained scopes, missing user → no scopes.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.assistant.runtime_context import RuntimeContext, build_runtime_for_user
from app.assistant.tools.base import ToolContext


# ── is_allowed / require ───────────────────────────────────────────────────


def test_wildcard_permission_grants_everything():
    rt = RuntimeContext.for_test(permissions=frozenset({"*"}))
    assert rt.is_allowed("memory:write")
    assert rt.is_allowed("genie:synthesize")
    assert rt.is_allowed("anything:at:all")


def test_specific_scope_grants_only_that_scope():
    rt = RuntimeContext.for_test(permissions=frozenset({"memory:write"}))
    assert rt.is_allowed("memory:write")
    assert not rt.is_allowed("memory:bulk_delete")
    assert not rt.is_allowed("genie:synthesize")


def test_require_raises_when_scope_missing():
    rt = RuntimeContext.for_test(permissions=frozenset())
    with pytest.raises(PermissionError):
        rt.require("memory:bulk_delete")


def test_require_passes_when_scope_present():
    rt = RuntimeContext.for_test(permissions=frozenset({"memory:bulk_delete"}))
    # Must not raise.
    rt.require("memory:bulk_delete")


# ── Frozenness ──────────────────────────────────────────────────────────────


def test_runtime_context_is_frozen():
    """Identity / scope cannot mutate mid-call. Frozen dataclass."""
    rt = RuntimeContext.for_test()
    with pytest.raises(Exception):
        rt.user_id = uuid.uuid4()  # type: ignore[misc]


# ── from_tool_context ───────────────────────────────────────────────────────


def test_from_tool_context_carries_identity_and_scope():
    uid = uuid.uuid4()
    sid = uuid.uuid4()
    pid = uuid.uuid4()
    ctx = ToolContext(
        user_id=uid,
        session_id=sid,
        namespace_key="cs.AI",
        namespace_keys=["cs.AI", "cs.NLP"],
        orientation="research",
        expertise_level="expert",
        job_id="job_xyz",
        parent_message_id=pid,
        db=MagicMock(),
        should_cancel=AsyncMock(return_value=False),
        emit_progress=AsyncMock(),
    )
    rt = RuntimeContext.from_tool_context(ctx, permissions=frozenset({"memory:write"}))
    assert rt.user_id == uid
    assert rt.session_id == sid
    assert rt.namespace_key == "cs.AI"
    assert rt.namespace_keys == ("cs.AI", "cs.NLP")
    assert rt.orientation == "research"
    assert rt.expertise_level == "expert"
    assert rt.job_id == "job_xyz"
    assert rt.request_id == "job_xyz"
    assert rt.parent_message_id == pid
    assert rt.is_allowed("memory:write")
    # Escape hatches surface the underlying ctx.
    assert rt.db is ctx.db
    assert rt.should_cancel is ctx.should_cancel


# ── build_runtime_for_user ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_runtime_grants_wildcard_for_admin():
    """An admin user gets ``*`` so ``is_allowed`` returns True for
    anything. Removes the need to enumerate every scope."""
    user = MagicMock()
    user.is_admin = True
    user.feature_overrides = {}
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    db.execute = AsyncMock(return_value=result)

    ctx = ToolContext(
        user_id=uuid.uuid4(), session_id=uuid.uuid4(),
        namespace_key="cs.AI", namespace_keys=["cs.AI"],
        orientation="both", expertise_level="practitioner",
        job_id="j", parent_message_id=uuid.uuid4(),
        db=MagicMock(), should_cancel=AsyncMock(), emit_progress=AsyncMock(),
    )
    rt = await build_runtime_for_user(db, ctx=ctx)
    assert rt.is_allowed("memory:write")
    assert rt.is_allowed("genie:synthesize")
    assert "*" in rt.permissions


@pytest.mark.asyncio
async def test_build_runtime_lifts_feature_overrides_to_scopes():
    """Truthy feature_overrides become fine-grained scopes — admins
    can grant per-capability access without flipping is_admin."""
    user = MagicMock()
    user.is_admin = False
    user.feature_overrides = {
        "memory:bulk_delete": True,
        "experimental:tool": True,
        "disabled:thing": False,  # falsy → NOT lifted
    }
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    db.execute = AsyncMock(return_value=result)

    ctx = ToolContext(
        user_id=uuid.uuid4(), session_id=uuid.uuid4(),
        namespace_key="cs.AI", namespace_keys=["cs.AI"],
        orientation="both", expertise_level="practitioner",
        job_id="j", parent_message_id=uuid.uuid4(),
        db=MagicMock(), should_cancel=AsyncMock(), emit_progress=AsyncMock(),
    )
    rt = await build_runtime_for_user(db, ctx=ctx)
    assert rt.is_allowed("memory:bulk_delete")
    assert rt.is_allowed("experimental:tool")
    assert not rt.is_allowed("disabled:thing")
    assert not rt.is_allowed("*")  # not admin


@pytest.mark.asyncio
async def test_build_runtime_safe_when_user_missing():
    """A vanished User row (race with delete) must not crash —
    runtime returns with empty permissions, the API boundary
    will already have rejected the request before this point."""
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)

    ctx = ToolContext(
        user_id=uuid.uuid4(), session_id=uuid.uuid4(),
        namespace_key="cs.AI", namespace_keys=["cs.AI"],
        orientation="both", expertise_level="practitioner",
        job_id="j", parent_message_id=uuid.uuid4(),
        db=MagicMock(), should_cancel=AsyncMock(), emit_progress=AsyncMock(),
    )
    rt = await build_runtime_for_user(db, ctx=ctx)
    assert rt.permissions == frozenset()
    assert not rt.is_allowed("memory:write")
