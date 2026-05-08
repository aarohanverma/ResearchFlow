"""Source registry — returns the correct BaseSource for a given source_name."""

from app.adapters.sources.base import BaseSource, RawPaper
from app.core.config import settings


class SourceRegistry:
    """Registry mapping source names to ``BaseSource`` implementation classes.

    Built-in sources (``arxiv_rss`` and ``arxiv_mcp``) are registered at
    module import time. Additional sources can be added via ``register``.
    """

    _registry: dict[str, type[BaseSource]] = {}

    @classmethod
    def register(cls, name: str, klass: type[BaseSource]) -> None:
        """Register a ``BaseSource`` subclass under the given name.

        Args:
            name: Source identifier string (e.g. ``"arxiv_rss"``).
            klass: The ``BaseSource`` subclass to register.
        """
        cls._registry[name] = klass

    @classmethod
    def get(cls, source_name: str) -> BaseSource:
        """Instantiate and return the source registered under ``source_name``.

        Args:
            source_name: Source identifier (e.g. ``"arxiv_rss"``).

        Returns:
            A fresh instance of the registered ``BaseSource`` subclass.

        Raises:
            ValueError: If ``source_name`` is not in the registry.
        """
        if source_name not in cls._registry:
            raise ValueError(f"Unknown source: {source_name}")
        return cls._registry[source_name]()


# Register built-in sources
from app.adapters.sources.arxiv_rss import ArXivRssSource
from app.adapters.sources.arxiv_mcp import ArXivMcpSource

SourceRegistry.register("arxiv_rss", ArXivRssSource)
SourceRegistry.register("arxiv_mcp", ArXivMcpSource)

__all__ = ["BaseSource", "RawPaper", "SourceRegistry"]
