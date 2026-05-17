"""Author Network tool — discover key researchers in a field.

Uses the OpenAlex Authors API (free, no key required) to find influential
researchers by name or field. Returns h-index, citation counts, paper counts,
affiliations, and top papers. Supports both name-based lookup ("Geoffrey Hinton")
and topic-based discovery ("transformer attention mechanisms").
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_OPENALEX_AUTHORS = "https://api.openalex.org/authors"
_OPENALEX_WORKS = "https://api.openalex.org/works"
_AUTHOR_SELECT = "id,display_name,cited_by_count,works_count,summary_stats,last_known_institution,x_concepts"
_MAILTO = "researchflow@example.com"
_TIMEOUT = 20.0

# Matches 1-4 title-case words (person name pattern)
_NAME_RE = re.compile(
    r"^[A-Z][a-zA-ZÀ-ÿ'\-\.]{1,30}"
    r"(?:\s+[A-Z]?[a-zA-ZÀ-ÿ'\-\.]{1,30}){0,3}$"
)


class AuthorNetworkInput(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description="Researcher name or topic to find key authors in (e.g. 'Yann LeCun' or 'transformer attention')",
    )
    limit: int = Field(default=6, ge=1, le=12)
    include_papers: bool = Field(default=True, description="Include each author's top cited papers")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class AuthorNetworkOutput(BaseModel):
    authors: list[dict]
    total: int


class AuthorNetworkTool:
    """Find influential researchers and their publication profiles via OpenAlex."""

    name = "author_network"
    summary = (
        "Discover key researchers in a field or look up a specific scientist's publication "
        "profile. Returns h-index, total citations, paper count, affiliation, and top papers. "
        "Use for: 'Who are the leading researchers in X?', 'What has Geoffrey Hinton published?', "
        "'Who should I follow in Y field?', 'Find experts in Z'. "
        "Powered by OpenAlex (200M+ works, free, no API key needed)."
    )
    cost_class = "moderate"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = AuthorNetworkInput
    output_schema = AuthorNetworkOutput

    async def run(self, ctx: ToolContext, params: AuthorNetworkInput) -> ToolResult:
        await ctx.emit_progress(10, f"Searching for researchers: {params.query[:60]}")

        is_name = bool(_NAME_RE.match(params.query.strip()))

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                if is_name:
                    authors = await _fetch_by_name(client, params.query, params.limit)
                else:
                    authors = await _fetch_by_topic(client, params.query, params.limit)

                await ctx.emit_progress(60, f"Enriching {len(authors)} researcher profiles...")

                if params.include_papers and authors:
                    for author in authors:
                        author["top_papers"] = await _fetch_top_papers(
                            client, author["author_id"]
                        )

        except Exception as exc:
            log.warning("author_network failed: %s", exc)
            return ToolResult(
                output={"authors": [], "total": 0},
                summary=f"Author search unavailable: {exc}",
            )

        if not authors:
            return ToolResult(
                output={"authors": [], "total": 0},
                summary=f"No researchers found for: {params.query}",
            )

        authors.sort(key=lambda a: a.get("citation_count") or 0, reverse=True)
        await ctx.emit_progress(100, f"Found {len(authors)} researchers")

        top = authors[0]
        return ToolResult(
            output={"authors": authors, "total": len(authors)},
            summary=(
                f"{len(authors)} researchers found "
                f"(top: {top['name']}, h-index={top.get('h_index')}, "
                f"{top.get('citation_count', 0):,} citations)"
            ),
        )


async def _fetch_by_name(
    client: httpx.AsyncClient, name: str, limit: int
) -> list[dict]:
    resp = await client.get(
        _OPENALEX_AUTHORS,
        params={
            "search": name,
            "per_page": min(limit * 2, 20),
            "select": _AUTHOR_SELECT,
            "mailto": _MAILTO,
        },
        headers={"User-Agent": "ResearchFlow/1.0"},
    )
    resp.raise_for_status()
    return [_shape_author(a) for a in (resp.json().get("results") or [])[:limit]]


async def _fetch_by_topic(
    client: httpx.AsyncClient, topic: str, limit: int
) -> list[dict]:
    """Find top authors by scanning highly-cited recent works on the topic."""
    current_year = datetime.now().year
    start_year = current_year - 5

    works_resp = await client.get(
        _OPENALEX_WORKS,
        params={
            "filter": f"abstract.search:{topic},publication_year:{start_year}-{current_year}",
            "sort": "cited_by_count:desc",
            "per_page": 50,
            "select": "id,cited_by_count,authorships",
            "mailto": _MAILTO,
        },
        headers={"User-Agent": "ResearchFlow/1.0"},
    )
    works_resp.raise_for_status()
    works = works_resp.json().get("results") or []

    # Weight each author by sum of citation counts of their contributing works
    author_score: dict[str, float] = {}
    for work in works:
        cite_weight = 1.0 + (work.get("cited_by_count") or 0) / 100.0
        for authorship in (work.get("authorships") or [])[:6]:
            aid = (authorship.get("author") or {}).get("id", "")
            if aid:
                author_score[aid] = author_score.get(aid, 0) + cite_weight

    if not author_score:
        return []

    # Extract short IDs (A12345) from full OpenAlex URLs
    top_full_ids = sorted(author_score, key=lambda k: author_score[k], reverse=True)
    short_ids = [uid.rstrip("/").split("/")[-1] for uid in top_full_ids[: limit * 2]]

    author_resp = await client.get(
        _OPENALEX_AUTHORS,
        params={
            "filter": "ids.openalex:" + "|".join(short_ids[:limit]),
            "per_page": limit,
            "select": _AUTHOR_SELECT,
            "mailto": _MAILTO,
        },
        headers={"User-Agent": "ResearchFlow/1.0"},
    )
    if author_resp.status_code != 200:
        log.warning("author batch fetch failed: %s", author_resp.status_code)
        return []

    return [_shape_author(a) for a in (author_resp.json().get("results") or [])[:limit]]


async def _fetch_top_papers(client: httpx.AsyncClient, author_id: str) -> list[dict]:
    """Fetch top 3 most-cited works for an author (short OpenAlex ID)."""
    if not author_id:
        return []
    try:
        resp = await client.get(
            _OPENALEX_WORKS,
            params={
                "filter": f"author.id:{author_id}",
                "sort": "cited_by_count:desc",
                "per_page": 3,
                "select": "id,title,publication_year,cited_by_count",
                "mailto": _MAILTO,
            },
            headers={"User-Agent": "ResearchFlow/1.0"},
        )
        if resp.status_code != 200:
            return []
        return [
            {
                "title": p.get("title", ""),
                "year": p.get("publication_year"),
                "citations": p.get("cited_by_count", 0),
            }
            for p in (resp.json().get("results") or [])[:3]
        ]
    except Exception:
        return []


def _shape_author(a: dict) -> dict:
    """Normalize an OpenAlex author record into the tool's output schema."""
    oa_url = a.get("id", "")
    short_id = oa_url.rstrip("/").split("/")[-1] if oa_url else ""
    inst = a.get("last_known_institution") or {}
    stats = a.get("summary_stats") or {}
    concepts = a.get("x_concepts") or []
    areas = [c.get("display_name", "") for c in concepts[:4] if c.get("display_name")]
    return {
        "author_id": short_id,
        "name": a.get("display_name", ""),
        "h_index": stats.get("h_index"),
        "i10_index": stats.get("i10_index"),
        "citation_count": a.get("cited_by_count", 0),
        "paper_count": a.get("works_count", 0),
        "affiliations": [inst["display_name"]] if inst.get("display_name") else [],
        "research_areas": areas,
        "openalex_url": oa_url,
        "top_papers": [],
        "source": "openalex_authors",
    }


author_network_tool = AuthorNetworkTool()
