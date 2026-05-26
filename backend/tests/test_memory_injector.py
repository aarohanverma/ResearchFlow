"""Tests for the query-aware memory injection middleware.

The middleware sits BEFORE the planner's prompt build and collapses
each memory tier down to its always-keep entries (preferences,
skill, procedure) plus the top-K query-relevant non-preference
entries. Properties this layer must hold:

  * Always-keep types survive regardless of semantic score — they
    drive ``_render_procedural_block`` and prompt-shaping rules and
    dropping them would silently change behaviour.
  * Small buckets pass through unchanged (no semantic call wasted).
  * When the embedder is offline the helper degrades to a recency-
    ordered slice; the planner never sees a missing-memory failure.
  * The helper never raises into the planner — the worst case is the
    original dump being passed through.
"""

from __future__ import annotations

import pytest

from app.assistant.memory_injector import (
    _ALWAYS_KEEP_TYPES,
    _DEFAULT_PER_TIER_K,
    select_relevant_memory,
)


def _entry(value: str, etype: str = "finding", ts: str = "2026-05-20T00:00:00Z") -> dict:
    return {"value": value, "type": etype, "ts": ts}


def _bucket(prefix: str, n: int, etype: str = "finding") -> dict:
    return {f"{prefix}_{i}": _entry(f"{prefix} number {i}", etype=etype) for i in range(n)}


# ── Pass-through paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pass_through_when_query_empty():
    """Empty query → no ranking signal → return as-is."""
    mv = {"short": _bucket("s", 20), "medium": {}, "long": {}}
    out = await select_relevant_memory(query="", memory_view=mv, session_id="sid")
    assert out["short"] == mv["short"]


@pytest.mark.asyncio
async def test_pass_through_when_bucket_already_small():
    """≤ per_tier_k non-preferences → no semantic call worth making."""
    small = _bucket("s", _DEFAULT_PER_TIER_K)   # exactly at cap
    mv = {"short": small, "medium": {}, "long": {}}
    out = await select_relevant_memory(query="anything", memory_view=mv, session_id="sid")
    assert out["short"] == small


@pytest.mark.asyncio
async def test_pass_through_non_dict_memory_view():
    """Defensive — non-dict in must come back unchanged."""
    out = await select_relevant_memory(
        query="x", memory_view="not a dict", session_id="sid",
    )  # type: ignore[arg-type]
    assert out == "not a dict"


# ── Always-keep types ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preferences_always_survive(monkeypatch):
    """Preferences MUST be preserved even when semantic ranking drops
    everything else — the planner's tone/depth shaping depends on
    them being present each turn."""
    prefs = {
        "tone_pref": _entry("prefer concise answers", etype="preference"),
        "depth_pref": _entry("prefer expert depth", etype="preference"),
    }
    others = _bucket("findings", 15)
    bucket = {**prefs, **others}
    mv = {"short": bucket, "medium": {}, "long": {}}

    # Force the semantic layer to return nothing — preferences should
    # still be in the output.
    async def _empty_rank(*, query, entries, session_id, top_k):
        return []

    monkeypatch.setattr(
        "app.assistant.memory_injector._semantic_rank_safe", _empty_rank,
    )
    out = await select_relevant_memory(query="q", memory_view=mv, session_id="sid")
    short = out["short"]
    assert "tone_pref" in short
    assert "depth_pref" in short


@pytest.mark.asyncio
async def test_procedurals_always_survive(monkeypatch):
    """``skill``/``procedure`` entries drive the planner's procedural
    block render — dropping them silently would remove user-saved
    instructions about HOW the agent behaves."""
    procs = {
        "always_cite": _entry("always cite arxiv ids", etype="procedure"),
        "math_format": _entry("render math in latex", etype="skill"),
    }
    others = _bucket("findings", 15)
    bucket = {**procs, **others}
    mv = {"short": {}, "medium": bucket, "long": {}}

    async def _empty_rank(*, query, entries, session_id, top_k):
        return []

    monkeypatch.setattr(
        "app.assistant.memory_injector._semantic_rank_safe", _empty_rank,
    )
    out = await select_relevant_memory(query="q", memory_view=mv, session_id="sid")
    medium = out["medium"]
    assert "always_cite" in medium
    assert "math_format" in medium


def test_always_keep_set_includes_known_types():
    """Lock in the always-keep contract so a future rename of the
    type literals doesn't silently change planner behaviour."""
    assert "preference" in _ALWAYS_KEEP_TYPES
    assert "skill" in _ALWAYS_KEEP_TYPES
    assert "procedure" in _ALWAYS_KEEP_TYPES


# ── Recency fallback when no session anchor ─────────────────────────────────


@pytest.mark.asyncio
async def test_recency_fallback_when_session_id_missing():
    """No session_id → no embedding cache anchor → take the recency
    fallback (newest entries first) without paying single-shot embed
    costs we'd discard."""
    bucket = {
        "old": _entry("old finding", ts="2026-01-01T00:00:00Z"),
        "mid": _entry("mid finding", ts="2026-03-01T00:00:00Z"),
        "new": _entry("new finding", ts="2026-06-01T00:00:00Z"),
    }
    # 15 entries so we exceed per_tier_k.
    for i in range(15):
        bucket[f"extra_{i}"] = _entry(f"extra {i}", ts="2026-02-01T00:00:00Z")
    mv = {"short": bucket, "medium": {}, "long": {}}

    out = await select_relevant_memory(
        query="anything", memory_view=mv, session_id=None,
    )
    short = out["short"]
    assert "new" in short, "newest entry should survive recency fallback"
    # Cap is honoured.
    assert len(short) <= _DEFAULT_PER_TIER_K


# ── Top-K query relevance ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_semantic_ranking_keeps_top_k(monkeypatch):
    """When the semantic layer ranks entries, the focused view keeps
    the highest-scored entries up to per_tier_k."""
    bucket = _bucket("paper_note", 20)
    mv = {"short": bucket, "medium": {}, "long": {}}

    # Fake rank: descending score, top entries are paper_note_0..K-1
    async def _fake_rank(*, query, entries, session_id, top_k):
        items = list(entries.items())[:top_k]
        return [(k, v, 1.0 - i * 0.01) for i, (k, v) in enumerate(items)]

    monkeypatch.setattr(
        "app.assistant.memory_injector._semantic_rank_safe", _fake_rank,
    )
    out = await select_relevant_memory(query="q", memory_view=mv, session_id="sid")
    short = out["short"]
    assert len(short) == _DEFAULT_PER_TIER_K
    for i in range(_DEFAULT_PER_TIER_K):
        assert f"paper_note_{i}" in short


# ── Never-raises contract ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_never_raises_when_semantic_layer_explodes(monkeypatch):
    """Any unexpected error in the semantic layer must NOT propagate
    into the planner — the helper must return the original memory_view
    so the planner still has something to render."""
    bucket = _bucket("s", 20)
    mv = {"short": bucket, "medium": {}, "long": {}}

    # Patch the underlying ranker — _semantic_rank_safe wraps it and
    # falls back to recency on any error.
    async def _broken_rank(*args, **kwargs):
        raise RuntimeError("embedder exploded")

    import app.assistant.semantic_memory as sm
    monkeypatch.setattr(sm, "semantically_rank", _broken_rank)

    out = await select_relevant_memory(
        query="q", memory_view=mv, session_id="sid",
    )
    # No raise; bucket still has at least the recency-fallback slice.
    assert "short" in out
    assert isinstance(out["short"], dict)
    assert len(out["short"]) <= _DEFAULT_PER_TIER_K
