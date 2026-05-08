"""CacheBackend ABC — local file or Redis, swapped by CACHE_BACKEND env var."""

from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    """Abstract base class for key-value cache backends.

    Two concrete implementations are provided: ``LocalFileCache`` for
    development (zero external dependencies) and ``RedisCache`` for
    production use on Azure.
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Return cached value or None."""

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store a value with optional TTL in seconds."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check existence without loading the value."""
