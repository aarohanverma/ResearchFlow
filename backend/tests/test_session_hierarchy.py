"""Tests for the session-hierarchy endpoint.

The user spec is explicit: from any session's perspective the
hierarchy view must show ancestors (root → ... → parent) plus
descendants (children, grandchildren, …) — but NEVER sibling
branches that share a parent but aren't on this session's line.

We also need cross-user safety: a leaked session id from another
user must 404 (existence not leaked).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.assistant import get_session_hierarchy


def _session(*, sid=None, parent=None, title="t", ns="cs.AI", branch_from=None):
    s = MagicMock()
    s.id = sid or uuid.uuid4()
    s.parent_session_id = parent
    s.title = title
    s.namespace_key = ns
    s.branch_from_message_id = branch_from
    s.created_at = MagicMock()  # not compared by tests
    return s


def _make_db_with_repo(*, current, ancestor_chain=None, children_map=None):
    """Wire up an AsyncMock db + repo where:
      - repo.get_session(user_id, sid) returns the session at sid;
      - db.execute(select children).scalars().all() returns the
        configured children list for each parent.
    """
    ancestor_chain = ancestor_chain or {}
    children_map = children_map or {}
    by_id = {current.id: current, **ancestor_chain}
    for parent_id, kids in children_map.items():
        for kid in kids:
            by_id[kid.id] = kid

    # Mock the AssistantRepository.get_session calls
    async def _get_session(user_id, sid):
        return by_id.get(sid)

    repo_mock = MagicMock()
    repo_mock.get_session = AsyncMock(side_effect=_get_session)

    db = AsyncMock()
    # db.execute for "children of X" queries; we infer the parent
    # from the side_effect counter by routing every execute call
    # through this stub. The order of execute calls matches BFS:
    # current -> direct children -> grandchildren, etc. We dequeue
    # the children for the next-popped node.
    bfs_targets = [current]
    visited_for_children = set()

    def _make_result(seq):
        r = MagicMock()
        r.scalars.return_value.all.return_value = seq
        return r

    async def _execute(_stmt):
        # Pop the next target node from the BFS frontier and return
        # its children. The BFS frontier is rebuilt as nodes are
        # processed.
        if bfs_targets:
            node = bfs_targets.pop(0)
            kids = list(children_map.get(node.id, []))
            visited_for_children.add(node.id)
            bfs_targets.extend(kids)
            return _make_result(kids)
        return _make_result([])

    db.execute = AsyncMock(side_effect=_execute)
    return db, repo_mock


@pytest.mark.asyncio
async def test_hierarchy_404_for_unknown_or_other_user_session(monkeypatch):
    """A session that doesn't belong to the caller (or doesn't
    exist) returns 404."""
    from fastapi import HTTPException

    me = uuid.uuid4()
    db = AsyncMock()
    repo_mock = MagicMock()
    repo_mock.get_session = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "app.api.v1.assistant.AssistantRepository",
        MagicMock(return_value=repo_mock),
    )

    with pytest.raises(HTTPException) as exc:
        await get_session_hierarchy(session_id=uuid.uuid4(), user_id=me, db=db)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_hierarchy_root_session_returns_full_descendant_tree(monkeypatch):
    """When the current session IS the root, the view returns every
    descendant. No ancestors are reported because there's nothing
    above the root."""
    me = uuid.uuid4()
    root = _session(title="root")
    branch_a = _session(parent=root.id, title="branch A")
    branch_b = _session(parent=root.id, title="branch B")
    leaf_a1 = _session(parent=branch_a.id, title="leaf A1")

    db, repo_mock = _make_db_with_repo(
        current=root,
        ancestor_chain={},
        children_map={
            root.id:     [branch_a, branch_b],
            branch_a.id: [leaf_a1],
            branch_b.id: [],
            leaf_a1.id:  [],
        },
    )
    monkeypatch.setattr(
        "app.api.v1.assistant.AssistantRepository",
        MagicMock(return_value=repo_mock),
    )

    out = await get_session_hierarchy(session_id=root.id, user_id=me, db=db)
    assert out["ancestors"] == []
    titles = [d["title"] for d in out["descendants"]]
    # All three descendants surface from the root view.
    assert "branch A" in titles
    assert "branch B" in titles
    assert "leaf A1" in titles
    assert out["counts"]["descendants"] == 3


@pytest.mark.asyncio
async def test_hierarchy_branch_view_excludes_sibling_branches(monkeypatch):
    """From a branch session's perspective, the hierarchy view must
    show:

      - its ancestor chain (root → this branch's parent),
      - itself,
      - its own descendants,

    But NOT sibling branches under the same parent. This is the
    user's explicit "child sees only its relevant hierarchy" rule.
    """
    me = uuid.uuid4()
    root = _session(title="root")
    branch_a = _session(parent=root.id, title="branch A")
    branch_b = _session(parent=root.id, title="branch B (sibling)")
    leaf_a1 = _session(parent=branch_a.id, title="leaf A1")

    db, repo_mock = _make_db_with_repo(
        current=branch_a,
        ancestor_chain={root.id: root},
        children_map={
            branch_a.id: [leaf_a1],
            leaf_a1.id:  [],
        },
    )
    monkeypatch.setattr(
        "app.api.v1.assistant.AssistantRepository",
        MagicMock(return_value=repo_mock),
    )

    out = await get_session_hierarchy(session_id=branch_a.id, user_id=me, db=db)
    # Root surfaces as an ancestor.
    ancestor_titles = [a["title"] for a in out["ancestors"]]
    assert "root" in ancestor_titles
    # Self is rendered as current.
    assert out["current"]["title"] == "branch A"
    assert out["current"]["is_current"] is True
    # Descendant of self surfaces.
    descendant_titles = [d["title"] for d in out["descendants"]]
    assert "leaf A1" in descendant_titles
    # Sibling under the same parent must NOT appear anywhere — this
    # is the load-bearing isolation guarantee.
    full_tree_titles = (
        ancestor_titles
        + [out["current"]["title"]]
        + descendant_titles
    )
    assert "branch B (sibling)" not in full_tree_titles


@pytest.mark.asyncio
async def test_hierarchy_self_only_when_no_relations(monkeypatch):
    """A standalone session (no parent, no children) returns
    just itself with empty ancestors and descendants."""
    me = uuid.uuid4()
    standalone = _session(title="alone")
    db, repo_mock = _make_db_with_repo(
        current=standalone,
        ancestor_chain={},
        children_map={standalone.id: []},
    )
    monkeypatch.setattr(
        "app.api.v1.assistant.AssistantRepository",
        MagicMock(return_value=repo_mock),
    )

    out = await get_session_hierarchy(session_id=standalone.id, user_id=me, db=db)
    assert out["ancestors"] == []
    assert out["descendants"] == []
    assert out["current"]["title"] == "alone"
