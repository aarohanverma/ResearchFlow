"""Redis-backed cache — used in Azure via Azure Cache for Redis."""

import json
from typing import Any

import redis.asyncio as aioredis

from app.adapters.cache.base import CacheBackend
from app.core.config import settings


class RedisCache(CacheBackend):
    """Redis-backed cache using the ``redis.asyncio`` client.

    Values are JSON-serialised before storage so that any JSON-compatible
    Python object can round-trip through the cache.
    """

    def __init__(self, url: str | None = None) -> None:
        """Initialise the async Redis client.

        Args:
            url: Redis connection URL. Falls back to ``settings.redis_url``
                if not provided.
        """
        self._client = aioredis.from_url(url or settings.redis_url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        """Return the deserialised value for the key, or ``None`` if absent.

        Args:
            key: Cache key string.

        Returns:
            The stored Python object, or ``None`` if the key does not exist.
        """
        raw = await self._client.get(key)
        return json.loads(raw) if raw is not None else None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Serialise and store a value in Redis with an optional TTL.

        Args:
            key: Cache key string.
            value: JSON-serialisable value to store.
            ttl_seconds: Seconds until expiry. ``None`` means no expiry.
        """
        serialized = json.dumps(value, default=str)
        if ttl_seconds:
            await self._client.setex(key, ttl_seconds, serialized)
        else:
            await self._client.set(key, serialized)

    async def delete(self, key: str) -> None:
        """Delete the key from Redis.

        Args:
            key: Cache key string to remove.
        """
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        """Return ``True`` if the key currently exists in Redis.

        Args:
            key: Cache key string to check.

        Returns:
            ``True`` if the key is present, ``False`` otherwise.
        """
        return bool(await self._client.exists(key))
