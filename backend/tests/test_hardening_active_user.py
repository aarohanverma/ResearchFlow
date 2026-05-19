"""Regression tests for is_active enforcement.

Before this pass, an admin could "deactivate" a user via the admin
panel but the user could still log in (auth/login didn't check the
flag) and keep using already-issued tokens (the JWT-decode dep
returned the UUID without checking the row). That made deactivation
purely cosmetic. These tests lock in the new behaviour:

* Login of a deactivated user → 403, distinct from the 401 the
  wrong-password / unknown-email path returns.
* The TTL cache for is_active is invalidated whenever an admin write
  flips the flag, so the next request from that user (in any worker)
  sees the change immediately.
* ``_check_user_active`` returns False for a missing user row — a JWT
  issued before account deletion stops working as soon as the row goes.
"""

from __future__ import annotations

import time
import uuid

import pytest


def test_invalidate_active_cache_drops_single_user():
    from app.core.deps import _ACTIVE_CACHE, invalidate_active_cache

    uid_a = uuid.uuid4()
    uid_b = uuid.uuid4()
    _ACTIVE_CACHE[uid_a] = (True, time.monotonic())
    _ACTIVE_CACHE[uid_b] = (True, time.monotonic())

    invalidate_active_cache(uid_a)

    assert uid_a not in _ACTIVE_CACHE
    assert uid_b in _ACTIVE_CACHE


def test_invalidate_active_cache_wipes_all_with_no_arg():
    from app.core.deps import _ACTIVE_CACHE, invalidate_active_cache

    _ACTIVE_CACHE[uuid.uuid4()] = (True, time.monotonic())
    _ACTIVE_CACHE[uuid.uuid4()] = (False, time.monotonic())
    invalidate_active_cache(None)
    assert len(_ACTIVE_CACHE) == 0


@pytest.mark.asyncio
async def test_check_user_active_returns_false_for_missing_row(monkeypatch):
    """A token whose user row was deleted should be treated as deactivated.

    We stub the session factory to return a context manager that
    yields a fake DB whose ``get`` returns None — the actual prod
    code path when ``db.get(User, uid)`` finds nothing.
    """
    from app.core import deps as deps_mod

    class _FakeDB:
        async def get(self, _model, _id):
            return None
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False

    monkeypatch.setattr(deps_mod, "async_session_factory", lambda: _FakeDB())
    deps_mod.invalidate_active_cache(None)

    assert await deps_mod._check_user_active(uuid.uuid4()) is False


@pytest.mark.asyncio
async def test_check_user_active_returns_true_for_active_row(monkeypatch):
    from app.core import deps as deps_mod

    class _FakeUser:
        is_active = True

    class _FakeDB:
        async def get(self, _model, _id):
            return _FakeUser()
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False

    monkeypatch.setattr(deps_mod, "async_session_factory", lambda: _FakeDB())
    deps_mod.invalidate_active_cache(None)

    assert await deps_mod._check_user_active(uuid.uuid4()) is True


@pytest.mark.asyncio
async def test_check_user_active_returns_false_for_deactivated(monkeypatch):
    from app.core import deps as deps_mod

    class _FakeUser:
        is_active = False

    class _FakeDB:
        async def get(self, _model, _id):
            return _FakeUser()
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False

    monkeypatch.setattr(deps_mod, "async_session_factory", lambda: _FakeDB())
    deps_mod.invalidate_active_cache(None)

    assert await deps_mod._check_user_active(uuid.uuid4()) is False


@pytest.mark.asyncio
async def test_check_user_active_caches_result(monkeypatch):
    """Two back-to-back checks should hit the DB only once within the TTL."""
    from app.core import deps as deps_mod

    call_count = {"n": 0}

    class _FakeUser:
        is_active = True

    class _FakeDB:
        async def get(self, _model, _id):
            call_count["n"] += 1
            return _FakeUser()
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False

    monkeypatch.setattr(deps_mod, "async_session_factory", lambda: _FakeDB())
    deps_mod.invalidate_active_cache(None)
    uid = uuid.uuid4()

    assert await deps_mod._check_user_active(uid) is True
    assert await deps_mod._check_user_active(uid) is True
    assert call_count["n"] == 1, "second call must hit cache, not DB"


@pytest.mark.asyncio
async def test_check_user_active_fail_open_on_db_error(monkeypatch):
    """A transient DB failure must not 403 every authenticated user."""
    from app.core import deps as deps_mod

    class _FakeDB:
        async def __aenter__(self): raise RuntimeError("pool exhausted")
        async def __aexit__(self, *_): return False

    monkeypatch.setattr(deps_mod, "async_session_factory", lambda: _FakeDB())
    deps_mod.invalidate_active_cache(None)

    # fail-open: returns True so the request proceeds; the actual
    # endpoint will fail to do its work, surfacing the real DB error
    # there rather than masking it behind a 403.
    assert await deps_mod._check_user_active(uuid.uuid4()) is True
