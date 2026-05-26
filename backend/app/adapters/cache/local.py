"""Local file-based cache — zero dependencies, works offline."""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import aiofiles

from app.adapters.cache.base import CacheBackend
from app.core.config import settings


class LocalFileCache(CacheBackend):
    """File-based cache backend that persists each entry as a JSON file.

    Suitable for local development — no external services required. Each
    cache entry is stored as ``<sanitized_key>.json`` under ``_dir``.
    Expired entries are lazily deleted on read.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        """Initialise the local file cache.

        Args:
            cache_dir: Root directory for cache files. Falls back to
                ``settings.cache_dir`` if not provided. The directory is
                created if it does not already exist.
        """
        self._dir = Path(cache_dir or settings.cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Return the filesystem ``Path`` for a given cache key.

        Sanitises the key for safe use as a filename. Keys longer than 180
        characters after sanitisation get a 16-char MD5 suffix so that two
        distinct long keys that share the same first 180 characters never map
        to the same file (collision risk under truncation-only approach).

        Args:
            key: Raw cache key string.

        Returns:
            Absolute ``Path`` to the JSON cache file for this key.
        """
        safe = key.replace("/", "_").replace(":", "_")
        if len(safe) > 180:
            digest = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:16]
            safe = safe[:160] + "_" + digest
        return self._dir / f"{safe}.json"

    async def get(self, key: str) -> Any | None:
        """Read a cached value, returning ``None`` if missing or expired.

        Args:
            key: Cache key string.

        Returns:
            The stored value, or ``None`` if the key does not exist or its
            TTL has elapsed.

        Notes:
            If the JSON payload is corrupted (truncated by a previous crash
            mid-write, or a non-atomic concurrent overwrite from another
            process), the corrupt file is deleted so the next ``set`` writes
            cleanly. Without this self-heal a corrupted entry persists as a
            permanent cache miss for the lifetime of the directory.
        """
        path = self._path(key)
        if not await asyncio.to_thread(path.exists):
            return None
        try:
            async with aiofiles.open(path) as f:
                raw = json.loads(await f.read())
            if raw.get("expires") and raw["expires"] < time.time():
                await self.delete(key)
                return None
            return raw.get("value")
        except (json.JSONDecodeError, ValueError):
            # Corrupted entry — evict so the next write is clean. Swallow
            # any unlink error since the only goal here is "don't return a
            # bad value"; we already return None on the line below.
            try:
                await asyncio.to_thread(path.unlink, missing_ok=True)
            except Exception:
                pass
            return None
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Write a value to the cache, optionally with a TTL.

        Args:
            key: Cache key string.
            value: JSON-serialisable value to store.
            ttl_seconds: Seconds until expiry. ``None`` means no expiry.

        Atomicity:
            Writes go to ``<path>.<pid>.tmp`` first, then ``os.replace`` it
            onto the final path. ``os.replace`` is atomic on POSIX +
            modern Windows, so a concurrent reader either sees the old file
            or the fully-written new file — never a truncated tail. A
            process killed mid-write leaves the ``.tmp`` sibling behind
            (cleaned up on next write attempt for the same key) but the
            cache entry stays consistent.
        """
        path = self._path(key)
        payload = {
            "value": value,
            "expires": time.time() + ttl_seconds if ttl_seconds else None,
        }
        # PID + monotonic-ns suffix so two concurrent writers on the SAME
        # key (different coroutines / processes) never collide on the
        # temp filename. The last replace() to land wins, which is the
        # right semantics for a cache.
        tmp_path = path.with_suffix(
            f"{path.suffix}.{os.getpid()}.{time.monotonic_ns()}.tmp"
        )
        try:
            async with aiofiles.open(tmp_path, "w") as f:
                await f.write(json.dumps(payload, default=str))
            await asyncio.to_thread(os.replace, str(tmp_path), str(path))
        except Exception:
            # Best-effort cleanup of the temp file on failure so the cache
            # directory doesn't accumulate .tmp leftovers.
            try:
                await asyncio.to_thread(
                    lambda: tmp_path.unlink(missing_ok=True),
                )
            except Exception:
                pass
            raise

    async def delete(self, key: str) -> None:
        """Delete the cache file for the given key, if it exists.

        Args:
            key: Cache key string to remove.
        """
        path = self._path(key)
        try:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        except Exception:
            pass

    async def exists(self, key: str) -> bool:
        """Return ``True`` if a non-expired value exists for the key.

        Args:
            key: Cache key string to check.

        Returns:
            ``True`` if the key is present and not expired, ``False`` otherwise.
        """
        return (await self.get(key)) is not None
