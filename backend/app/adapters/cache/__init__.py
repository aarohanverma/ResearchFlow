from app.adapters.cache.base import CacheBackend
from app.core.config import settings


def get_cache() -> CacheBackend:
    """Instantiate and return the configured cache backend.

    Returns:
        A ``RedisCache`` instance when ``settings.cache_backend`` is
        ``"redis"``; otherwise a ``LocalFileCache`` instance.
    """
    if settings.cache_backend == "redis":
        from app.adapters.cache.redis_cache import RedisCache
        return RedisCache()
    from app.adapters.cache.local import LocalFileCache
    return LocalFileCache()


__all__ = ["CacheBackend", "get_cache"]
