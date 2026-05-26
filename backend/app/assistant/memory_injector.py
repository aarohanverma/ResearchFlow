"""Query-aware memory injection middleware.

Before each model call, the planner used to receive a flat dump of every
stored entry across the short / medium / long memory tiers. As memory
accumulates, that dump bloats the system prompt with entries that are
unrelated to the user's current question — wasting context, biasing
plans toward unrelated topics, and slowing the cheap-model decision
call.

This module provides the missing middleware step that the LangChain
multi-agent guides call out: *automatic, query-aware retrieval of the
top-relevant memories, with only a compact slice injected into the
prompt*. The architecture is:

  Store (DB + session.state)   ← memory_recall tool writes / explicit ops
        │
        ▼
  Memory injector (THIS MODULE)  ← runs before each planner call
        │  per tier:
        │    * always keep preferences (small, load-bearing for tone /
        │      depth); they don't gain from semantic ranking
        │    * rank everything else by semantic similarity to the user
        │      query (uses existing ``semantically_rank``)
        │    * fall back to recency when the embedder is offline so
        │      behavior degrades gracefully, never breaks
        ▼
  Planner / ReAct decision prompt

Design notes:

  * **Strictly additive** — when the embedder is unavailable, every
    tier returns a recency-sorted slice with at-least the original
    preferences preserved. Existing callers that pass the unfiltered
    memory_view still work; this helper is opt-in.
  * **No store changes** — the persistent layer (session.state +
    semantic embedding cache) is untouched. We only filter what we
    inject at prompt-render time.
  * **No removal of expressed agency** — the existing
    ``memory_recall``/``memory_write``/``memory_delete`` tools remain
    the explicit, agent-controlled path. This middleware is the
    automatic, background-level convenience that complements (does not
    replace) them.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

log = logging.getLogger(__name__)


# Per-tier non-preference cap. Tuned conservatively: the planner already
# weights preferences separately, so the rest of the budget goes to
# query-relevant context. Preferences are appended on top and are not
# counted against this cap.
_DEFAULT_PER_TIER_K = 6


def _entry_type(v: object) -> str:
    if isinstance(v, dict):
        return str(v.get("type", "context"))
    return "context"


# Entry types that should always survive filtering regardless of
# semantic similarity to the current query:
#   * ``preference`` — load-bearing for tone / depth / format
#   * ``skill`` / ``procedure`` — drive ``_render_procedural_block``;
#     dropping them would silently remove user-saved instructions
#     about HOW the agent should behave.
_ALWAYS_KEEP_TYPES: frozenset[str] = frozenset({
    "preference", "skill", "procedure",
})


def _is_always_keep(v: object) -> bool:
    return _entry_type(v) in _ALWAYS_KEEP_TYPES


# Backwards-compat for tests written against the original helper name.
def _is_preference(v: object) -> bool:
    return _entry_type(v) == "preference"


def _entry_ts(v: object) -> str:
    if isinstance(v, dict):
        return str(v.get("ts") or v.get("updated_at") or "")
    return ""


async def _semantic_rank_safe(
    *,
    query: str,
    entries: dict[str, dict],
    session_id: UUID | str | None,
    top_k: int,
) -> list[tuple[str, dict, float]]:
    """Wrap ``semantically_rank`` so any failure falls back to recency.

    The semantic layer is best-effort: if the embedder errors out, the
    network is offline, or session_id is missing (test contexts), we
    return a recency-ordered slice with score 0 so callers can still
    pick the top-K. The middleware NEVER raises into the planner.
    """
    if not query or not entries:
        return []
    if not session_id:
        # No session anchor → can't cache embeddings; fall back to
        # recency directly to avoid the cost of single-shot embeds we'd
        # discard anyway.
        items = sorted(
            entries.items(),
            key=lambda kv: _entry_ts(kv[1]),
            reverse=True,
        )
        return [(k, v, 0.0) for k, v in items[:top_k]]
    try:
        from app.assistant.semantic_memory import semantically_rank
        return await semantically_rank(
            query=query,
            entries=entries,
            session_id=session_id,
            top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001 — degrade silently
        log.debug("memory_injector: semantic rank failed: %s", exc)
        items = sorted(
            entries.items(),
            key=lambda kv: _entry_ts(kv[1]),
            reverse=True,
        )
        return [(k, v, 0.0) for k, v in items[:top_k]]


async def select_relevant_memory(
    *,
    query: str,
    memory_view: dict[str, Any],
    session_id: UUID | str | None,
    per_tier_k: int = _DEFAULT_PER_TIER_K,
) -> dict[str, Any]:
    """Return a focused, query-relevant slice of ``memory_view``.

    Args:
        query: The current user query — drives semantic ranking.
        memory_view: The flat dict the orchestrator built, with keys
            ``"short"``, ``"medium"``, ``"long"``. Other keys
            (e.g. ``"injection_enabled"``) are passed through unchanged.
        session_id: Anchors the semantic embedding cache so embeddings
            from prior turns are reused. ``None`` is allowed (tests,
            cold starts) and falls back to recency.
        per_tier_k: Maximum number of NON-preference entries kept per
            tier after ranking. Preferences are kept in full (their
            small count + universal usefulness for tone/depth makes
            filtering them counterproductive).

    Returns:
        A new dict shaped like ``memory_view`` whose ``short``,
        ``medium``, ``long`` buckets contain only the preferences plus
        the top-K query-relevant non-preference entries. When a bucket
        already has ≤ ``per_tier_k`` non-preferences, it is returned
        as-is (no point ranking when everything fits).

    Notes:
        Pass-through guarantees:
            * Empty / missing buckets stay empty / missing.
            * Buckets that are already small (≤ ``per_tier_k``
              non-preferences) are returned unchanged — no semantic
              call is made.
            * When the embedder is unavailable, the focused view is a
              recency-ordered slice of the same shape. Callers cannot
              distinguish a degraded path from an empty bucket beyond
              the order of entries.

        This function NEVER raises. Worst-case, it returns the
        original ``memory_view`` unchanged so the planner still has
        something to render.
    """
    if not isinstance(memory_view, dict):
        return memory_view  # type: ignore[return-value]

    out: dict[str, Any] = dict(memory_view)
    if not query:
        # Without a query we can't rank — keep the original dump.
        return out

    for tier in ("short", "medium", "long"):
        raw = memory_view.get(tier) or {}
        if not isinstance(raw, dict) or not raw:
            continue
        always_keep = {k: v for k, v in raw.items() if _is_always_keep(v)}
        others = {k: v for k, v in raw.items() if not _is_always_keep(v)}
        if len(others) <= per_tier_k:
            # Already compact — no win from semantic ranking.
            continue
        ranked = await _semantic_rank_safe(
            query=query,
            entries=others,
            session_id=session_id,
            top_k=per_tier_k,
        )
        focused = dict(always_keep)
        for key, entry, _score in ranked:
            focused[key] = entry
        out[tier] = focused
    return out


__all__ = ["select_relevant_memory"]
