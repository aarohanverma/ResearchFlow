"""GitHub search tool — find code repositories related to research papers.

Searches GitHub for repositories implementing specific algorithms, models, or
research concepts. Uses the GitHub Search API (free, 60 req/hr unauthenticated,
5000 req/hr with GITHUB_TOKEN). Returns repos with star counts, languages,
descriptions, and last-updated timestamps.

Use when the user wants to find code implementations of a research concept, paper,
or algorithm — complements arXiv/Semantic Scholar for the implementation side of
research.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_TIMEOUT = 15.0
_GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
_MAX_RESULTS = 10

# Stop-words removed during compress — function words that GitHub's lexical
# matcher will down-rank or ignore. Kept small and English-only since the
# tool itself targets english repositories.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
    "to", "for", "with", "from", "by", "is", "are", "was", "were", "be",
    "being", "been", "have", "has", "had", "do", "does", "did", "i", "me",
    "my", "you", "your", "we", "us", "our", "this", "that", "those", "these",
    "it", "its", "as", "so", "than", "then", "there", "here", "what", "which",
    "who", "whom", "how", "why", "when", "where", "should", "would", "could",
    "can", "may", "might", "will", "shall", "any", "some", "all", "no", "not",
    "find", "show", "give", "want", "need", "please", "make", "build",
    "implementation", "implement", "implementations", "code", "codes",
    "research", "paper", "papers", "project", "projects",
})


def _compress_keyword_query(raw: str, *, max_words: int = 8) -> str:
    """Compress a verbose natural-language query into a tight keyword phrase.

    GitHub's search API matches by lexical overlap on a handful of terms; a
    long sentence like "find me research code implementation for retrieval
    augmented generation with chunk size and top-k retrieval" reliably
    returns zero results. We strip stop-words and keep the most informative
    domain terms (longer + non-stopword), capped at ``max_words``.

    Pass-through behaviour when the input is already concise (≤6 words) so
    we don't accidentally over-trim legitimate compact queries.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        return cleaned
    words = cleaned.split()
    if len(words) <= 6:
        return cleaned

    # Keep words that are NOT stop-words OR are short technical tokens
    # (e.g. "RAG", "LLM", "FFT"). Preserve original casing where possible
    # since GitHub treats e.g. "PyTorch" and "pytorch" as the same token
    # for matching but the user sees the query echoed in logs.
    informative: list[str] = []
    for w in words:
        token = w.strip(".,;:?!\"'()[]{}")
        if not token:
            continue
        if token.lower() in _STOPWORDS:
            continue
        informative.append(token)
        if len(informative) >= max_words:
            break

    # If filtering stripped too aggressively, fall back to the first
    # ``max_words`` words verbatim so we still send something.
    if len(informative) < 3:
        informative = words[:max_words]
    return " ".join(informative)


class GitHubSearchInput(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=120,
        description=(
            "SHORT keyword phrase naming the technique or library, NOT a "
            "sentence. GitHub's index uses keyword matching, not semantic "
            "search. Good: 'retrieval augmented generation pytorch', "
            "'mixture of experts jax', 'GraphSAGE pytorch geometric'. "
            "Bad: 'find me research code for RAG with chunk size and top-k "
            "retrieval' (too verbose, will return 0 results). Keep to "
            "3–8 concrete terms."
        ),
    )
    language: str = Field(default="", description="Filter by programming language (e.g. 'python', 'jupyter notebook')")
    min_stars: int = Field(default=0, description="Minimum star count filter (e.g. 50 for established repos)")
    sort: str = Field(default="stars", description="Sort by: 'stars', 'forks', 'updated', 'best-match'")
    max_results: int = Field(default=8, ge=1, le=_MAX_RESULTS)
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class GitHubSearchOutput(BaseModel):
    total_count: int
    repositories: list[dict]
    query_used: str


class GitHubSearchTool:
    """Search GitHub for code repositories implementing research concepts."""

    name = "github_search"
    summary = (
        "Search GitHub for code repositories implementing a research algorithm, model, or concept. "
        "Use when: user asks about code implementations, wants to find reference implementations, "
        "or is looking for practical examples of a research technique. "
        "Returns repos with stars, language, description, and GitHub URL. "
        "DO NOT use for finding papers — use arxiv_search or deep_search for that."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = GitHubSearchInput
    output_schema = GitHubSearchOutput

    async def run(self, ctx: ToolContext, params: GitHubSearchInput) -> ToolResult:
        await ctx.emit_progress(20, f"Searching GitHub for '{params.query[:60]}'…")

        # GitHub's search ranks by lexical overlap and chokes on long
        # phrases. If the planner sent a verbose sentence (over ~8 words),
        # compress it to the most-informative content keywords before
        # querying — this is how we turn 'find me retrieval augmented
        # generation chunk size top-k research implementation code' into
        # something GitHub actually matches.
        q = _compress_keyword_query(params.query)
        if params.language:
            q += f" language:{params.language}"
        if params.min_stars > 0:
            q += f" stars:>={params.min_stars}"

        sort = params.sort if params.sort in ("stars", "forks", "updated", "best-match") else "stars"

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ResearchFlow/1.0",
        }

        # Use GITHUB_TOKEN if available for higher rate limits
        try:
            from app.core.config import get_settings
            settings = get_settings()
            token = getattr(settings, "github_token", None) or ""
            if token:
                headers["Authorization"] = f"Bearer {token}"
        except Exception:
            pass

        repos: list[dict] = []
        total_count = 0

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _GITHUB_SEARCH_URL,
                    headers=headers,
                    params={
                        "q": q,
                        "sort": sort if sort != "best-match" else None,
                        "order": "desc",
                        "per_page": params.max_results,
                    },
                )

                if resp.status_code == 403:
                    return ToolResult(
                        output={"total_count": 0, "repositories": [], "query_used": q},
                        summary="GitHub API rate limit reached. Try again in a minute.",
                    )

                resp.raise_for_status()
                data = resp.json()
                total_count = data.get("total_count", 0)

                for item in data.get("items", [])[:params.max_results]:
                    updated_at = item.get("updated_at", "")
                    try:
                        updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        updated_str = updated_dt.strftime("%Y-%m")
                    except Exception:
                        updated_str = updated_at[:7] if updated_at else ""

                    repos.append({
                        "name": item.get("full_name", ""),
                        "description": (item.get("description") or "")[:200],
                        "url": item.get("html_url", ""),
                        "stars": item.get("stargazers_count", 0),
                        "forks": item.get("forks_count", 0),
                        "language": item.get("language") or "",
                        "topics": item.get("topics", [])[:5],
                        "updated": updated_str,
                        "license": (item.get("license") or {}).get("spdx_id", ""),
                    })

        except httpx.HTTPStatusError as exc:
            log.warning("github_search: HTTP error %s for query '%s'", exc.response.status_code, q)
            return ToolResult(
                output={"total_count": 0, "repositories": [], "query_used": q},
                summary=f"GitHub search failed (HTTP {exc.response.status_code})",
            )
        except Exception as exc:
            log.warning("github_search: error for query '%s': %s", q, exc)
            return ToolResult(
                output={"total_count": 0, "repositories": [], "query_used": q},
                summary=f"GitHub search failed: {exc}",
            )

        await ctx.emit_progress(100, f"Found {len(repos)} repositories")

        top_names = ", ".join(r["name"] for r in repos[:3])
        return ToolResult(
            output={
                "total_count": total_count,
                "repositories": repos,
                "query_used": q,
            },
            summary=(
                f"GitHub: {total_count:,} total repos for '{params.query[:50]}'. "
                f"Top results: {top_names or 'none found'}"
            ),
        )


github_search_tool = GitHubSearchTool()
