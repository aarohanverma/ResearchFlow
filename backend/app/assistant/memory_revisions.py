"""Audit-trail recorder for long-term memory writes.

Every mutation of a tracked memory entry (auto-write, manual write,
overwrite, delete, restore, supersession) lands in the
``memory_revisions`` table via :func:`record_revision`. The live state
remains on :class:`AssistantSession.state` JSONB — this table is the
append-only log behind the Settings → Memory inspect / restore UI.

Design notes:

* **Best-effort by design.** Recording is wrapped in try/except so a
  schema-drift or transient DB error can never block the live memory
  write the user actually asked for. We log the failure and continue —
  the live state stays intact, the user just won't see this revision
  in their history.

* **User-scoped.** Every row carries ``user_id`` so the inspect
  endpoints can query directly without joining through
  ``assistant_sessions``. A leaked session id can never surface
  another user's history.

* **Class-aware.** We derive the cognitive-science class
  (semantic / episodic / procedural / preference / "-") from the
  memory type via :func:`memory_category` so the UI can filter by
  the three-way taxonomy the user explicitly asked for.

* **Subject/topic enrichment.** ``namespace_key`` (e.g. ``cs.AI``)
  is split into ``subject`` (``cs``) and ``topic`` (``AI``) so the
  UI can filter at a finer grain than the namespace alone.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.tools.memory import memory_category
from app.models.assistant import AssistantSession, MemoryRevision

log = logging.getLogger(__name__)


_VALID_ACTIONS: frozenset[str] = frozenset({
    "create", "update", "delete", "restore", "supersede",
})


def split_subject_topic(namespace_key: str) -> tuple[str, str]:
    """Decompose a namespace key into ``(subject, topic)``.

    Namespace keys follow the convention ``subject.topic`` (e.g.
    ``cs.AI``, ``physics.quant-ph``). When the key has no separator,
    the entire string is treated as the subject and the topic is
    empty — keeps the audit row well-formed even for legacy
    single-word namespaces.
    """
    nk = (namespace_key or "").strip()
    if not nk:
        return "", ""
    if "." not in nk:
        return nk[:60], ""
    subject, _, topic = nk.partition(".")
    return subject[:60], topic[:60]


async def _resolve_root_session(
    db: AsyncSession, session_id: UUID | str,
) -> AssistantSession | None:
    """Walk to the root session of a chain.

    Bounded at 20 hops to match the rest of the memory pipeline's
    cycle guard. Falls back to ``None`` if the session row vanished
    (race with delete) so the caller can skip the audit row rather
    than crash the live write.
    """
    seen: set = set()
    current = await db.get(AssistantSession, session_id)
    if current is None:
        return None
    for _ in range(20):
        if current.parent_session_id is None or current.parent_session_id in seen:
            return current
        seen.add(current.id)
        parent = await db.get(AssistantSession, current.parent_session_id)
        if parent is None:
            return current
        current = parent
    return current


async def record_revision(
    db: AsyncSession,
    *,
    user_id: UUID | str,
    session_id: UUID | str | None,
    tier: str,
    key: str,
    value: str,
    action: str,
    namespace_key: str = "",
    entry_type: str = "context",
    source: str = "manual",
    previous_value: str | None = None,
    status: str = "active",
    confidence: float | None = None,
    ttl_days: int | None = None,
    extras: dict[str, Any] | None = None,
) -> bool:
    """Append one revision row to the audit log.

    Returns ``True`` if the row was added, ``False`` if recording was
    skipped (malformed input or transient DB error). Never raises —
    the caller's live write must succeed regardless of audit-log
    state.

    The function does NOT commit; the caller decides transaction
    boundaries so revision + live state can land atomically when
    desired. Auto-memory consolidation already commits its own
    session, so calling this inside that session is the common
    pattern.

    Args:
        user_id: Owner of the entry.
        session_id: Originating session — the chat where the write
            happened. Used to resolve the root session that owns
            the memory bucket. May be ``None`` for system-initiated
            restores; in that case the caller must supply
            ``extras["root_session_id"]``.
        tier: ``short`` / ``medium`` / ``long``.
        key: Normalised memory key.
        value: New value being written (empty string for deletes).
        action: One of ``create`` / ``update`` / ``delete`` /
            ``restore`` / ``supersede``.
        namespace_key: For ``long``-tier writes, the namespace bucket.
            Empty for short/medium.
        entry_type: The content-shape label
            (``finding`` / ``preference`` / etc.).
        source: ``manual`` (user-driven), ``auto`` (consolidation),
            ``restore`` (restoring an earlier revision).
        previous_value: For updates/deletes/restores, the value prior
            to this revision. Enables diff/compare in the UI.
        status: The entry's status at the time of this revision.
        confidence: Optional [0, 1] confidence the writer attached.
        ttl_days: Per-entry TTL if any.
        extras: Free-form JSONB — origin metadata, telemetry, etc.
    """
    if action not in _VALID_ACTIONS:
        log.debug("memory_revisions: refusing unknown action %r", action)
        return False
    if not key or not isinstance(key, str):
        log.debug("memory_revisions: refusing empty/non-str key")
        return False
    if tier not in {"short", "medium", "long"}:
        log.debug("memory_revisions: refusing unknown tier %r", tier)
        return False

    try:
        # Resolve the root session that owns the bucket. The audit
        # log indexes on root_session_id so multi-root users can be
        # filtered cleanly.
        root_id: UUID | None = None
        if session_id is not None:
            root = await _resolve_root_session(db, session_id)
            if root is not None:
                root_id = root.id
        if root_id is None:
            forced = (extras or {}).get("root_session_id")
            if forced:
                root_id = forced if isinstance(forced, UUID) else UUID(str(forced))
        if root_id is None:
            log.debug(
                "memory_revisions: no root session resolved for "
                "user=%s tier=%s key=%s — skipping audit row",
                user_id, tier, key,
            )
            return False

        subject, topic = split_subject_topic(namespace_key)
        cls = memory_category(entry_type) or "-"
        merged_extras = dict(extras or {})
        # Always surface the class in extras so retrieval / filtering
        # paths that don't read the dedicated column still see it.
        merged_extras.setdefault("memory_class", cls)

        row = MemoryRevision(
            user_id=user_id if isinstance(user_id, UUID) else UUID(str(user_id)),
            root_session_id=root_id,
            origin_session_id=(
                session_id if isinstance(session_id, UUID)
                else (UUID(str(session_id)) if session_id else None)
            ),
            tier=tier,
            namespace_key=(namespace_key or "")[:120],
            subject=subject,
            topic=topic,
            key=key[:200],
            value=(value or "")[:8000],
            previous_value=(previous_value[:8000] if isinstance(previous_value, str) else None),
            entry_type=(entry_type or "context")[:40],
            source=(source or "manual")[:40],
            action=action,
            status=(status or "active")[:20],
            confidence=confidence,
            ttl_days=ttl_days,
            extras=merged_extras,
        )
        # ``AsyncSession.add`` is a SYNC method in SQLAlchemy. Some
        # tests pass an ``AsyncMock`` for the whole db where ``add``
        # is incorrectly async; if we get a coroutine back we await
        # it so the call completes and pytest doesn't warn about a
        # never-awaited coroutine. Production code paths are
        # unaffected — the sync ``.add()`` returns ``None``.
        result = db.add(row)
        import inspect
        if inspect.iscoroutine(result):
            await result
        return True
    except Exception as exc:  # noqa: BLE001 — audit must never crash live write
        log.warning(
            "memory_revisions: failed to record %s for user=%s key=%s: %s",
            action, user_id, key, exc,
        )
        return False


def derive_entry_status(
    entry: dict[str, Any] | str | None,
    *,
    now_iso: str | None = None,
) -> str:
    """Compute the user-visible status for a live memory entry.

    Returns one of ``active`` / ``stale``. ``superseded`` and
    ``deleted`` are only ever observed in the revision history — a
    live entry that's still in the bucket is by definition not
    superseded or deleted.

    Stale = the entry has a ``ttl_days`` and the window expired.
    Mirrors :func:`app.assistant.tools.memory._memory_is_stale` to
    keep behavior in sync without crossing module imports at runtime.
    """
    from app.assistant.tools.memory import _memory_is_stale
    if _memory_is_stale(entry, now_iso=now_iso):
        return "stale"
    return "active"


__all__ = [
    "derive_entry_status",
    "record_revision",
    "split_subject_topic",
]
