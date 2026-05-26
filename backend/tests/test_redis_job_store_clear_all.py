"""RedisJobStore.clear_all must use SCAN, not KEYS.

``KEYS *`` is O(N) and blocks the Redis server for the full duration of
the scan — in a multi-tenant Redis instance a dev-reset call can stall
every concurrent request. ``SCAN`` streams keys in bounded batches and
never holds the server for a full pass.

This test installs a fake redis client that records every call and
asserts:
  * ``clear_all`` calls ``scan_iter``, NEVER ``keys``.
  * Every job key under the prefix is deleted.
  * User-index keys are also cleared (the operator wants a clean slate).
  * Deletes happen in bounded batches (no single megacall under load).
"""

from __future__ import annotations

import pytest

from app.services.job_store import RedisJobStore


class _RecordingFakeRedis:
    """Minimal async fake — records calls and yields keys via scan_iter."""

    def __init__(self, keys: list[str]) -> None:
        self._keys = list(keys)
        self.calls: list[tuple[str, tuple, dict]] = []
        self.deleted: list[str] = []

    async def scan_iter(self, match: str | None = None, count: int | None = None):
        self.calls.append(("scan_iter", (), {"match": match, "count": count}))
        for k in list(self._keys):
            if match is None or _glob_match(k, match):
                yield k

    async def keys(self, pattern: str):
        self.calls.append(("keys", (pattern,), {}))
        return [k for k in self._keys if _glob_match(k, pattern)]

    async def delete(self, *args: str) -> int:
        self.calls.append(("delete", args, {}))
        for k in args:
            self.deleted.append(k)
            if k in self._keys:
                self._keys.remove(k)
        return len(args)


def _glob_match(key: str, pattern: str) -> bool:
    if not pattern.endswith("*"):
        return key == pattern
    return key.startswith(pattern[:-1])


@pytest.mark.asyncio
async def test_clear_all_uses_scan_iter_not_keys(monkeypatch):
    store = RedisJobStore(redis_url="redis://noop")
    fake = _RecordingFakeRedis(
        keys=[
            f"rf:jobs:{i:03d}" for i in range(450)
        ] + [
            f"rf:jobs:user:u-{i}" for i in range(10)
        ] + [
            "unrelated:other:key",
        ]
    )

    async def _fake_get_client() -> "_RecordingFakeRedis":
        return fake

    monkeypatch.setattr(store, "_get_client", _fake_get_client)

    await store.clear_all()

    call_names = [c[0] for c in fake.calls]
    assert "scan_iter" in call_names, "clear_all must use SCAN, not KEYS"
    assert "keys" not in call_names, "KEYS blocks Redis — must not appear"

    # Both prefixes were scanned.
    scan_patterns = {c[2].get("match") for c in fake.calls if c[0] == "scan_iter"}
    assert "rf:jobs:*" in scan_patterns
    assert "rf:jobs:user:*" in scan_patterns

    # Every job key was deleted; the unrelated key was not.
    assert all(f"rf:jobs:{i:03d}" in fake.deleted for i in range(450))
    assert all(f"rf:jobs:user:u-{i}" in fake.deleted for i in range(10))
    assert "unrelated:other:key" not in fake.deleted

    # Deletes are batched — no single megacall.
    max_batch = max(
        len(c[1]) for c in fake.calls if c[0] == "delete"
    )
    assert max_batch <= 200, f"batch too large: {max_batch}"


@pytest.mark.asyncio
async def test_clear_all_noop_when_client_unavailable(monkeypatch):
    """Redis unavailable → silent return, no exception (dev-reset path
    must remain idempotent and unforgiving operational conditions)."""
    store = RedisJobStore(redis_url="redis://noop")

    async def _none_client() -> None:
        return None

    monkeypatch.setattr(store, "_get_client", _none_client)
    await store.clear_all()  # no raise
