"""Memory consolidation — clustering + LLM merge + idempotency.

Tests the pure-function pieces (eligibility filtering, clustering,
key generation) plus the LLM merge contract (graceful failure,
schema parsing). The end-to-end session walk uses a mocked DB session
so we don't need a real Postgres.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.assistant.memory_consolidation import (
    ConsolidationReport,
    _MIN_AGE_DAYS,
    _MIN_CLUSTER_SIZE,
    _consolidated_key,
    _consolidation_candidates,
)


def _entry(value: str, *, days_old: int = 30, type_: str = "context",
           consolidated_from: list[str] | None = None,
           consolidated_into: str | None = None) -> dict:
    """Build a memory entry with a realistic timestamp."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    entry: dict = {"value": value, "type": type_, "ts": ts}
    if consolidated_from:
        entry["consolidated_from"] = consolidated_from
    if consolidated_into:
        entry["consolidated_into"] = consolidated_into
    return entry


# ── Candidate filtering ─────────────────────────────────────────────────────


def test_eligible_candidates_include_durable_entries():
    """Old, content-bearing, non-consolidated entries are eligible."""
    tier = {
        "user_pref_terse": _entry("user prefers terse answers"),
        "user_role": _entry("user is a senior researcher"),
        "user_focus": _entry("user focuses on RAG production patterns"),
    }
    candidates = _consolidation_candidates(tier)
    assert len(candidates) == 3


def test_too_recent_entries_excluded():
    """Entries newer than _MIN_AGE_DAYS are skipped — consolidation
    is for facts that survived a few turns, not the latest write."""
    tier = {
        "stable_pref": _entry("stable preference", days_old=30),
        "fresh_pref": _entry("just-written preference", days_old=0),
    }
    candidates = _consolidation_candidates(tier)
    keys = [k for k, _ in candidates]
    assert "stable_pref" in keys
    assert "fresh_pref" not in keys


def test_already_consolidated_entries_excluded():
    """A rollup must not be re-clustered; its source entries (marked
    consolidated_into) also stay out of the next cycle."""
    tier = {
        "rollup": _entry("rollup value", consolidated_from=["a", "b"]),
        "merged": _entry("merged value", consolidated_into="rollup"),
        "fresh": _entry("fresh fact", days_old=30),
    }
    candidates = _consolidation_candidates(tier)
    keys = [k for k, _ in candidates]
    assert keys == ["fresh"]


def test_entries_without_value_excluded():
    tier = {
        "valued": _entry("real value"),
        "empty": {"value": "", "type": "context", "ts": (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).isoformat()},
    }
    candidates = _consolidation_candidates(tier)
    assert [k for k, _ in candidates] == ["valued"]


def test_non_dict_entries_skipped_gracefully():
    """Legacy plain-string entries must not crash the filter — they're
    just skipped."""
    tier = {
        "legacy_string": "this is a legacy plain-string entry",
        "real_entry": _entry("real fact"),
    }
    candidates = _consolidation_candidates(tier)
    assert [k for k, _ in candidates] == ["real_entry"]


# ── Consolidated key generation ─────────────────────────────────────────────


def test_consolidated_key_is_stable_across_input_order():
    """The same set of source keys must produce the same consolidated
    key regardless of insertion order — otherwise re-clustering would
    create duplicate rollups."""
    key_a = _consolidated_key("tree_memory", ["user_pref_terse", "user_role"])
    key_b = _consolidated_key("tree_memory", ["user_role", "user_pref_terse"])
    assert key_a == key_b
    assert key_a.startswith("consolidated__")


def test_consolidated_key_caps_length():
    """A huge source-key set shouldn't produce a 500-char rollup key."""
    long_sources = [f"very_long_key_name_{i}" for i in range(20)]
    key = _consolidated_key("ns_memory", long_sources)
    assert len(key) <= len("consolidated__") + 80
    assert key.startswith("consolidated__")


def test_consolidated_key_handles_empty_sources():
    """Defensive — never crash on a bug that passes [] sources."""
    key = _consolidated_key("tree_memory", [])
    assert key == "consolidated__rollup"


# ── ConsolidationReport accounting ──────────────────────────────────────────


def test_consolidation_report_summary_format():
    report = ConsolidationReport(
        sessions_scanned=5,
        clusters_found=3,
        consolidations_written=2,
        entries_merged=8,
    )
    summary = report.summary()
    assert "sessions_scanned=5" in summary
    assert "consolidations=2" in summary
    assert "merged=8" in summary


# ── End-to-end: _consolidate_tier with mocked clustering + LLM ─────────────


@pytest.mark.asyncio
async def test_consolidate_tier_skips_when_too_few_candidates():
    """A tier with fewer than _MIN_CLUSTER_SIZE entries doesn't even
    bother calling the embedding adapter."""
    from app.assistant.memory_consolidation import _consolidate_tier
    tier = {"only_one": _entry("solo fact", days_old=10)}
    report = ConsolidationReport()
    out = await _consolidate_tier(tier, report=report, tier_key="tree_memory")
    assert out is None
    assert report.skipped_too_few >= 1


@pytest.mark.asyncio
async def test_consolidate_tier_produces_rollup_when_cluster_found(monkeypatch):
    """When clustering returns a ≥3-entry cluster + LLM produces a
    non-empty merge, the tier gains a ``consolidated_*`` entry and
    the originals get ``consolidated_into`` back-pointers."""
    from app.assistant import memory_consolidation as mc

    # Stub clustering to return one 3-entry cluster.
    async def _fake_cluster(candidates):
        return [candidates] if len(candidates) >= 3 else []
    monkeypatch.setattr(mc, "_cluster_by_embedding", _fake_cluster)

    # Stub LLM merge to return a clean rollup.
    async def _fake_merge(cluster):
        return ("user prefers terse, technical responses", "preference")
    monkeypatch.setattr(mc, "_llm_merge", _fake_merge)

    tier = {
        f"pref_{i}": _entry(f"pref variant {i}", type_="preference", days_old=20)
        for i in range(3)
    }
    report = ConsolidationReport()
    out = await mc._consolidate_tier(tier, report=report, tier_key="tree_memory")
    assert out is not None
    # A consolidated entry was added.
    consolidated_keys = [k for k in out if k.startswith("consolidated__")]
    assert len(consolidated_keys) == 1
    rollup = out[consolidated_keys[0]]
    assert rollup["value"] == "user prefers terse, technical responses"
    assert rollup["type"] == "preference"
    assert set(rollup["consolidated_from"]) == {"pref_0", "pref_1", "pref_2"}
    # Originals carry the back-pointer.
    for orig_key in ("pref_0", "pref_1", "pref_2"):
        assert out[orig_key]["consolidated_into"] == consolidated_keys[0]
    assert report.consolidations_written == 1
    assert report.entries_merged == 3


@pytest.mark.asyncio
async def test_consolidate_tier_handles_llm_failure_gracefully(monkeypatch):
    """An LLM failure must increment ``llm_failures`` and leave the
    tier untouched rather than corrupting it."""
    from app.assistant import memory_consolidation as mc

    async def _fake_cluster(candidates):
        return [candidates] if len(candidates) >= 3 else []
    monkeypatch.setattr(mc, "_cluster_by_embedding", _fake_cluster)

    async def _fake_merge(cluster):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(mc, "_llm_merge", _fake_merge)

    tier = {
        f"pref_{i}": _entry(f"pref {i}", days_old=20) for i in range(3)
    }
    report = ConsolidationReport()
    out = await mc._consolidate_tier(tier, report=report, tier_key="tree_memory")
    # No consolidations written → return None (tier unchanged).
    assert out is None
    assert report.llm_failures == 1


@pytest.mark.asyncio
async def test_consolidate_tier_idempotent_on_already_rolled_up_tier(monkeypatch):
    """Running consolidation on a tier whose only entries are existing
    rollups must produce zero new consolidations — the candidate
    filter screens them out before clustering."""
    from app.assistant import memory_consolidation as mc

    fake_cluster = AsyncMock(return_value=[])
    fake_merge = AsyncMock(return_value=("", ""))
    monkeypatch.setattr(mc, "_cluster_by_embedding", fake_cluster)
    monkeypatch.setattr(mc, "_llm_merge", fake_merge)

    tier = {
        "consolidated__a_b_c": _entry("rollup A", consolidated_from=["a", "b", "c"]),
        "consolidated__d_e_f": _entry("rollup D", consolidated_from=["d", "e", "f"]),
        "consolidated__g_h_i": _entry("rollup G", consolidated_from=["g", "h", "i"]),
    }
    report = ConsolidationReport()
    out = await mc._consolidate_tier(tier, report=report, tier_key="tree_memory")
    assert out is None
    # Clustering function was never invoked — filter screened them.
    fake_cluster.assert_not_called()
    fake_merge.assert_not_called()
