"""BaseSource contract — RSS and MCP implementations both conform to this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawPaper:
    """Normalized paper from any source before DB persistence."""
    external_id: str         # arXiv ID
    title: str
    authors: list[str]
    abstract: str
    source_url: str
    pdf_url: str | None
    published_at: datetime | None
    namespace_key: str       # e.g. cs.AI
    raw: dict = field(default_factory=dict)   # original payload for debugging


class BaseSource(ABC):
    """Abstract base class for paper ingestion sources.

    Concrete implementations (e.g. ``ArXivRssSource``, ``ArXivMcpSource``)
    must set ``source_name`` and implement ``fetch``.

    Attributes:
        source_name: Identifier matching ``SourceMapping.source_name`` in the
            database (e.g. ``"arxiv_rss"``).
    """

    source_name: str         # matches SourceMapping.source_name

    @abstractmethod
    async def fetch(self, external_category_key: str) -> list[RawPaper]:
        """Fetch papers for the given category. Rate-limited internally."""

    async def normalize(self, papers: list[RawPaper]) -> list[RawPaper]:
        """Optional normalization pass — default is identity."""
        return papers

    async def dedupe(self, papers: list[RawPaper], existing_ids: set[str]) -> list[RawPaper]:
        """Remove papers already in the DB by external_id."""
        return [p for p in papers if p.external_id not in existing_ids]
