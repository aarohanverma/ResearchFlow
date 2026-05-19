"""Per-step result caching keyed on (tool, params, namespace, [user]).

Wraps the platform's existing CacheBackend (local-file or Redis depending on
``CACHE_BACKEND``). Cache hits short-circuit tool execution while still
writing an AssistantStep row so the reasoning tree shows a "cache hit" entry.

Two cache-key shapes:

* **User-scoped** (default): ``assistant:step:{tool}:{user}:{ns}:{hash}``
  — for tools whose output depends on user-private state (bookmarks,
  memory, interest profile).
* **Shared** : ``assistant:step:{tool}:shared:{ns}:{hash}`` — for tools
  that hit public/deterministic sources (arXiv, Wikipedia, Wolfram,
  Semantic Scholar, etc.). Dropping the user prefix lets every user
  reuse the same cached result, which is exactly the dedup goal for
  shared deterministic outputs.

Membership in ``_SHARED_SOURCE_TOOLS`` is the explicit allow-list. Anything
not listed stays user-scoped, so adding a new tool defaults to safe
behaviour.
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


# Tools whose output is a pure function of (params, public corpus) — no
# dependence on the calling user's private state. These share their cache
# across every user, so identical params dedup to a single fetch+compute.
_SHARED_SOURCE_TOOLS: frozenset[str] = frozenset({
    # arXiv / academic search & retrieval
    "arxiv_search",
    "arxiv_import",
    "pubmed",
    "inspire_hep",
    "nasa_ads",
    "crossref",
    "citation_finder",
    "literature_survey",
    "research_trends",
    "author_network",
    "semantic_scholar",
    "compare_papers",
    "frontier_scan",
    # Knowledge bases
    "wikipedia",
    "concept_explain",
    "wolfram_alpha",
    "oeis",
    # Code / model search
    "github_search",
    "huggingface_search",
    "papers_with_code",
    # Security / clinical / economic
    "nvd_cve",
    "clinicaltrials",
    "fred",
    # Misc deterministic
    "unpaywall",
    "latex_parse",
    "web_search",
    "deep_search",
    "graph_query",
    "graph_neighbors",
})


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
        """Stable cache key.

        Shared-source tools (``_SHARED_SOURCE_TOOLS``) drop the user
        segment so every user reuses a single cached result, which is
        the dedup goal for deterministic public-data tools. All other
        tools stay user-scoped to avoid leaking private-state-dependent
        outputs (e.g. bookmarks queries).
        """
        canonical = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        scope = "shared" if tool_name in _SHARED_SOURCE_TOOLS else str(user_id)
        return f"assistant:step:{tool_name}:{scope}:{namespace_key}:{digest}"

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
