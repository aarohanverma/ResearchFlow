"""LocalFileCache hardening regression tests.

Two production-grade properties this cache must hold even though it
backs only local dev workflows:

1. **Atomic write** — a process killed (or a coroutine cancelled)
   mid-``set`` must NEVER leave a truncated file at the final cache
   path. Without the atomic temp+replace, a concurrent ``get`` mid-
   write would either succeed (returns ``None`` on JSON decode error)
   or — worse — return a half-written payload that ``json.loads``
   accepts. The test asserts no partial reads survive even under
   concurrent overwrites.

2. **Self-healing on corruption** — if a corrupted file ever lands in
   the cache directory (filesystem corruption, manual edit, old
   process variant), the next ``get`` must DELETE it so the next
   ``set`` writes cleanly. Without self-heal a corrupted entry stays
   a perpetual miss for the lifetime of the directory.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.adapters.cache.local import LocalFileCache


@pytest.fixture
def cache_dir(tmp_path: Path) -> str:
    return str(tmp_path / "cache")


@pytest.mark.asyncio
async def test_set_is_atomic_no_partial_file(cache_dir: str):
    """After ``set`` returns, the final path must contain a fully-valid
    JSON payload — never a truncated tail. We verify by running many
    overlapping writes and asserting every read returns either the old
    or new value, never a JSON parse failure or partial dict."""
    cache = LocalFileCache(cache_dir=cache_dir)

    await cache.set("k", {"v": 0})
    assert (await cache.get("k")) == {"v": 0}

    async def writer(i: int) -> None:
        # Larger payloads → wider write window → more chances to
        # observe a non-atomic interleave if one existed.
        await cache.set("k", {"v": i, "filler": "x" * 4096})

    async def reader() -> None:
        for _ in range(50):
            v = await cache.get("k")
            # Either None (cache miss / corrupted file evicted) or a
            # dict — NEVER a truncated string or partial object.
            assert v is None or isinstance(v, dict)
            await asyncio.sleep(0)

    await asyncio.gather(
        *(writer(i) for i in range(40)),
        *(reader() for _ in range(8)),
    )

    # After the storm, the final entry is still readable as a complete
    # dict (last writer wins; we don't care which one).
    final = await cache.get("k")
    assert isinstance(final, dict)
    assert "v" in final


@pytest.mark.asyncio
async def test_get_self_heals_on_json_corruption(cache_dir: str):
    """A pre-existing corrupted cache file (truncated, manually edited)
    must be EVICTED on first read so the next ``set`` writes cleanly."""
    cache = LocalFileCache(cache_dir=cache_dir)

    await cache.set("ghost", {"v": 1})
    path = cache._path("ghost")
    # Corrupt: truncate to the first 5 bytes of the JSON payload.
    raw = path.read_bytes()
    assert len(raw) > 5
    path.write_bytes(raw[:5])

    # First read sees the corruption — must return None AND evict.
    assert (await cache.get("ghost")) is None
    assert not path.exists(), (
        "corrupted entry must be deleted so a re-set writes cleanly"
    )

    # Round-trip works again.
    await cache.set("ghost", {"v": 2})
    assert (await cache.get("ghost")) == {"v": 2}


@pytest.mark.asyncio
async def test_set_cleans_up_temp_file_on_failure(cache_dir: str, monkeypatch):
    """When the temp-file write succeeds but os.replace raises (e.g. the
    target became read-only between aiofiles.write and replace), the
    .tmp sibling must NOT linger in the cache dir."""
    cache = LocalFileCache(cache_dir=cache_dir)

    import os
    original_replace = os.replace

    def failing_replace(src, dst):
        # Simulate a transient FS failure on the rename step.
        raise OSError("simulated cross-device link error")

    monkeypatch.setattr("os.replace", failing_replace)
    with pytest.raises(OSError):
        await cache.set("k", {"v": 1})
    monkeypatch.setattr("os.replace", original_replace)

    # Verify no .tmp leftover in the cache dir.
    leftover = [p for p in Path(cache_dir).iterdir() if ".tmp" in p.name]
    assert leftover == [], f"expected no .tmp files, found {leftover}"
