"""Memory write-gating: freshness / version / conflict detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.assistant.tools.memory import _memory_is_stale


def _entry(value: str, ts: datetime, ttl_days: int | None = None) -> dict:
    e = {"value": value, "type": "finding", "ts": ts.isoformat()}
    if ttl_days is not None:
        e["ttl_days"] = ttl_days
    return e


def test_evergreen_entries_never_stale():
    """An entry without a TTL is evergreen — it never goes stale on
    its own (e.g. a user preference, a name, a definition)."""
    old = _entry("user prefers terse answers", datetime.now(timezone.utc) - timedelta(days=400))
    assert _memory_is_stale(old) is False


def test_fresh_ttl_entry_not_stale():
    fresh = _entry(
        "recent finding",
        datetime.now(timezone.utc) - timedelta(days=2),
        ttl_days=7,
    )
    assert _memory_is_stale(fresh) is False


def test_expired_ttl_entry_is_stale():
    expired = _entry(
        "old finding",
        datetime.now(timezone.utc) - timedelta(days=10),
        ttl_days=7,
    )
    assert _memory_is_stale(expired) is True


def test_legacy_string_entry_not_flagged_stale():
    """Pre-versioning legacy entries that were stored as plain strings
    must not crash the stale check — they're treated as evergreen."""
    assert _memory_is_stale("legacy string entry") is False


def test_bad_ttl_value_treated_as_evergreen():
    """A corrupted ``ttl_days`` field must not crash the gate; we fall
    back to treating the entry as evergreen."""
    bad = {
        "value": "weird entry",
        "ts": (datetime.now(timezone.utc) - timedelta(days=400)).isoformat(),
        "ttl_days": "not-a-number",
    }
    assert _memory_is_stale(bad) is False


def test_missing_timestamp_treated_as_evergreen():
    """No timestamp → no way to compute age → treat as evergreen rather
    than auto-flagging every untimestamped entry as stale."""
    no_ts = {"value": "x", "ttl_days": 7}
    assert _memory_is_stale(no_ts) is False
