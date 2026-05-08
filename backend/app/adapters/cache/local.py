"""Local file-based cache — zero dependencies, works offline."""

import asyncio
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
        """Return the filesystem ``Path`` for a given cache key, sanitizing it for safe filenames."""
        safe = key.replace("/", "_").replace(":", "_")[:200]
        return self._dir / f"{safe}.json"

    async def get(self, key: str) -> Any | None:
        """Read a cached value, returning ``None`` if missing or expired.

        Args:
            key: Cache key string.

        Returns:
            The stored value, or ``None`` if the key does not exist or its
            TTL has elapsed.
        """
        path = self._path(key)
        if not path.exists():
            return None
        try:
            async with aiofiles.open(path) as f:
                raw = json.loads(await f.read())
            if raw.get("expires") and raw["expires"] < time.time():
                await self.delete(key)
                return None
            return raw.get("value")
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Write a value to the cache, optionally with a TTL.

        Args:
            key: Cache key string.
            value: JSON-serialisable value to store.
            ttl_seconds: Seconds until expiry. ``None`` means no expiry.
        """
        path = self._path(key)
        payload = {
            "value": value,
            "expires": time.time() + ttl_seconds if ttl_seconds else None,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(payload, default=str))

    async def delete(self, key: str) -> None:
        """Delete the cache file for the given key, if it exists.

        Args:
            key: Cache key string to remove.
        """
        path = self._path(key)
        try:
            path.unlink(missing_ok=True)
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
