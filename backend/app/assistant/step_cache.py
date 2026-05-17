"""Per-step result caching keyed on (tool, params, namespace, user).

Wraps the platform's existing CacheBackend (local-file or Redis depending on
``CACHE_BACKEND``). Cache hits short-circuit tool execution while still
writing an AssistantStep row so the reasoning tree shows a "cache hit" entry.

Each tool declares its own TTL via ``cache_ttl_seconds``; tools that aren't
cacheable (side_effects=True, or anything time-sensitive) skip the cache
entirely. The cache is keyed on a stable hash of the input params so a
reorder of dict keys doesn't fork the cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import UUID

from app.adapters.cache import get_cache

log = logging.getLogger(__name__)


# Tool-specific TTLs (seconds). Tools not listed default to ``_DEFAULT_TTL``;
# tools with ``side_effects=True`` are never cached regardless.
_TOOL_TTL: dict[str, int] = {
    # Internal corpus tools — change slowly within a session
    "deep_search": 3600,          # 1 hour
    "frontier_scan": 1800,        # 30 min — arXiv frontier can update daily
    "graph_query": 3600,          # 1 hour — graph structure changes slowly
    "graph_neighbors": 1800,
    "bookmarks_query": 900,       # 15 min — user bookmarks can change
    # External academic search — stable for hours
    "arxiv_search": 3600,         # 1 hour — arXiv papers are immutable
    "arxiv_import": 1800,         # 30 min
    "pubmed": 7200,               # 2 hours — PubMed records are very stable
    "inspire_hep": 7200,          # 2 hours
    "nasa_ads": 7200,             # 2 hours
    "crossref": 7200,             # 2 hours — bibliographic metadata is immutable
    "citation_finder": 3600,      # 1 hour
    "literature_survey": 3600,    # 1 hour
    "research_trends": 14400,     # 4 hours — publication counts change slowly
    "author_network": 7200,       # 2 hours
    # Knowledge / computation
    "wikipedia": 3600,            # 1 hour — Wikipedia changes slowly
    "concept_explain": 1800,      # 30 min
    "wolfram_alpha": 86400,       # 24 hours — math results are deterministic
    "oeis": 86400,                # 24 hours — OEIS is read-only reference data
    # Code / model search
    "github_search": 1800,        # 30 min — repos change; rankings shift
    "huggingface_search": 1800,
    "papers_with_code": 3600,     # 1 hour — benchmarks stable
    # Security and clinical — stable within a day
    "nvd_cve": 3600,              # 1 hour — CVE records rarely change once published
    "clinicaltrials": 3600,       # 1 hour
    # Economic data — updated infrequently
    "fred": 14400,                # 4 hours — FRED data releases are scheduled
    # Unpaywall — DOI-stable open-access links
    "unpaywall": 86400,           # 24 hours — OA status changes rarely
}
_DEFAULT_TTL = 600  # 10 minutes default for tools not listed above


class StepCache:
    """Cache key construction + read/write around the existing CacheBackend."""

    def __init__(self) -> None:
        self._backend = get_cache()

    def is_cacheable(self, tool) -> bool:
        """A tool is cacheable iff it's pure (no side effects)."""
        return not getattr(tool, "side_effects", True)

    def make_key(
        self,
        *,
        tool_name: str,
        params: dict[str, Any],
        user_id: UUID,
        namespace_key: str,
    ) -> str:
        """Stable cache key. User-scoped to avoid cross-user data bleed."""
        canonical = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return f"assistant:step:{tool_name}:{user_id}:{namespace_key}:{digest}"

    async def get(self, key: str) -> dict | None:
        """Return cached step output or ``None`` on miss."""
        try:
            return await self._backend.get(key)
        except Exception as exc:
            log.warning("step cache get failed key=%s: %s", key, exc)
            return None

    async def set(self, key: str, value: dict, *, tool_name: str) -> None:
        """Persist a step output with the tool's TTL."""
        ttl = _TOOL_TTL.get(tool_name, _DEFAULT_TTL)
        try:
            await self._backend.set(key, value, ttl_seconds=ttl)
        except Exception as exc:
            log.warning("step cache set failed key=%s: %s", key, exc)


_CACHE: StepCache | None = None


def get_step_cache() -> StepCache:
    """Singleton accessor — built lazily so test patching works."""
    global _CACHE
    if _CACHE is None:
        _CACHE = StepCache()
    return _CACHE
