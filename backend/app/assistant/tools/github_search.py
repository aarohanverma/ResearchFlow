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


class GitHubSearchInput(BaseModel):
    query: str = Field(description="Search query for GitHub repositories (e.g. 'transformer attention mechanism pytorch')")
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

        q = params.query.strip()
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
