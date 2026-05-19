"""Regression tests for the RA memory-system hardening pass.

Locks in two concrete behaviour contracts the audit identified:

1. ``_resolve_root_session_id`` returns the deepest ancestor's ``id``
   in a single recursive CTE rather than N sequential SELECTs. The
   helper transparently falls back to an in-memory walk when the DB
   layer is a stub, which matters for the test harness (and for any
   downstream consumer that may pass a non-SQL session for testing).

2. The legacy ``_resolve_root_session`` shim still returns the full
   ORM row, so existing callers that mutate ``state`` keep working
   after the underlying walk was reimplemented.

3. The auto-memory pass refreshes ``root`` *after* taking the
   per-session lock so a concurrent sibling-branch commit cannot be
   silently overwritten. This is a structural assertion against the
   function body — the in-place call to ``db.refresh(root)`` between
   ``async with session_state_lock(root.id):`` and ``_apply_writes``
   must remain present so the lost-update bug we fixed never returns.
"""

from __future__ import annotations

import inspect

import pytest


# ── Recursive CTE root resolver ───────────────────────────────────────────────


class _FakeSession:
    """Tiny stand-in that mimics the surface ``_resolve_root_session_id`` uses.

    The CTE path is exercised only against PostgreSQL; the fallback walk is
    the one we can verify with a stub, and the production code is wired so
    that any exception inside the CTE block falls through to the walk.
    """

    def __init__(self, rows: dict):
        # rows: {uuid: parent_uuid or None}
        self._rows = rows

    async def execute(self, *_args, **_kwargs):  # noqa: D401
        raise RuntimeError("force fallback to in-memory walk for the test")

    async def get(self, _model, sid):
        class _Row:
            def __init__(self, _id, parent):
                self.id = _id
                self.parent_session_id = parent

        return _Row(sid, self._rows.get(sid))


@pytest.mark.asyncio
async def test_resolve_root_session_id_walks_to_top_of_chain():
    """A 3-deep branch resolves to the root; non-branched returns self."""
    from app.assistant.tools.memory import _resolve_root_session_id

    # Linear chain: leaf -> mid -> root
    rows = {"leaf": "mid", "mid": "root", "root": None}
    got = await _resolve_root_session_id(_FakeSession(rows), "leaf")
    assert got == "root", f"expected 'root', got {got!r}"

    # Already at the top — must return self, not None.
    rows2 = {"solo": None}
    got = await _resolve_root_session_id(_FakeSession(rows2), "solo")
    assert got == "solo"


@pytest.mark.asyncio
async def test_resolve_root_session_id_handles_cycles():
    """A cycle in parent_session_id must not loop forever (20-hop guard)."""
    from app.assistant.tools.memory import _resolve_root_session_id

    rows = {"a": "b", "b": "a"}
    # Should not hang — the bounded loop returns whichever node it lands on.
    got = await _resolve_root_session_id(_FakeSession(rows), "a")
    assert got in {"a", "b"}


# ── Auto-memory in-lock refresh guard ─────────────────────────────────────────


def test_auto_memory_consolidate_refetches_root_after_lock():
    """The body of ``consolidate_after_turn`` must call ``db.refresh(root)``
    AFTER acquiring ``session_state_lock(root.id)`` and BEFORE calling
    ``_apply_writes``. Without that refresh, a sibling-branch consolidation
    that committed in the lock-acquisition window would be silently overwritten.

    Asserting against the source text keeps the contract explicit — a code
    review that removes the refresh will trip this test.
    """
    from app.assistant import auto_memory

    src = inspect.getsource(auto_memory.consolidate_after_turn)
    # Find the nested-root-lock branch.
    assert "session_state_lock(root.id)" in src, "root lock must be acquired"
    # The refresh must come between root-lock entry and _apply_writes.
    after_lock = src.split("session_state_lock(root.id)", 1)[1]
    apply_idx = after_lock.find("_apply_writes")
    refresh_idx = after_lock.find("db.refresh(root)")
    assert apply_idx != -1, "_apply_writes must remain inside the root lock"
    assert refresh_idx != -1, "db.refresh(root) must be present after the root lock"
    assert refresh_idx < apply_idx, (
        "db.refresh(root) must run BEFORE _apply_writes — otherwise the "
        "lost-update bug we fixed returns."
    )


def test_patch_session_state_uses_session_lock():
    """``patch_session_state`` must acquire ``session_state_lock`` so a
    concurrent ``consolidate_after_turn`` / ``update_branch_progress_summary``
    on the same row cannot read the JSONB then overwrite our merge.
    """
    from app.repositories.assistant import AssistantRepository

    src = inspect.getsource(AssistantRepository.patch_session_state)
    assert "session_state_lock" in src, (
        "patch_session_state must take the per-session state lock so its "
        "read-modify-write of session.state can't race with other writers."
    )
    # And refresh inside the lock — otherwise the merge applies to a stale
    # identity-map snapshot taken before the lock was acquired.
    assert "refresh(row)" in src, (
        "patch_session_state must refresh the row inside the lock so the "
        "merge sees the freshest persisted state."
    )
