"""Research Trends tool — publication growth analysis via OpenAlex.

Uses the OpenAlex API (free, no key required) to show year-by-year publication
volume for a research topic. Useful for answering: "Is X a growing area?",
"When did interest in Y take off?", "How active is Z research right now?"

Also surfaces top venues (journals/conferences) and co-occurring concepts.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_OPENALEX_WORKS = "https://api.openalex.org/works"
_OPENALEX_CONCEPTS = "https://api.openalex.org/concepts"
_TIMEOUT = 15.0
_MAILTO = "researchflow@example.com"


class ResearchTrendsInput(BaseModel):
    topic: str = Field(min_length=2, max_length=300, description="Research topic, keyword, or concept to analyze")
    years_back: int = Field(default=8, ge=2, le=20, description="How many years of history to include")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class ResearchTrendsOutput(BaseModel):
    topic: str
    yearly_counts: list[dict]  # [{year, count}]
    top_venues: list[dict]     # [{name, count}]
    trend: str                 # "growing", "declining", "stable", "unknown"
    peak_year: int | None
    total_recent: int


class ResearchTrendsTool:
    """Analyze publication growth trends for a research topic over time."""

    name = "research_trends"
    summary = (
        "Analyze year-by-year publication volume for a research topic using OpenAlex "
        "(200M+ papers). Shows whether a field is growing or declining, identifies the "
        "peak year of activity, surfaces top publication venues (journals/conferences), "
        "and reveals emerging vs. established areas. "
        "Use for: 'Is X a hot area?', 'When did attention mechanisms become popular?', "
        "'What are the top venues for Y research?', 'Is Z field still active?'"
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = ResearchTrendsInput
    output_schema = ResearchTrendsOutput

    async def run(self, ctx: ToolContext, params: ResearchTrendsInput) -> ToolResult:
        await ctx.emit_progress(15, f"Analyzing trends for: {params.topic[:60]}")

        current_year = datetime.now().year
        start_year = current_year - params.years_back

        yearly_counts: list[dict] = []
        top_venues: list[dict] = []

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                # Year-by-year breakdown via group_by
                resp = await client.get(
                    _OPENALEX_WORKS,
                    params={
                        "filter": f"abstract.search:{params.topic},publication_year:{start_year}-{current_year}",
                        "group_by": "publication_year",
                        "mailto": _MAILTO,
                    },
                    headers={"User-Agent": "ResearchFlow/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()

                for group in (data.get("group_by") or []):
                    try:
                        year = int(group.get("key", 0))
                        count = int(group.get("count", 0))
                        if year >= start_year:
                            yearly_counts.append({"year": year, "count": count})
                    except (ValueError, TypeError):
                        continue

                yearly_counts.sort(key=lambda x: x["year"])

                await ctx.emit_progress(55, "Fetching top venues...")

                # Top venues for this topic
                venue_resp = await client.get(
                    _OPENALEX_WORKS,
                    params={
                        "filter": f"abstract.search:{params.topic},publication_year:{current_year - 5}-{current_year}",
                        "group_by": "primary_location.source.display_name",
                        "per_page": 10,
                        "mailto": _MAILTO,
                    },
                    headers={"User-Agent": "ResearchFlow/1.0"},
                )
                if venue_resp.status_code == 200:
                    venue_data = venue_resp.json()
                    for group in (venue_data.get("group_by") or [])[:8]:
                        name = group.get("key_display_name") or group.get("key", "")
                        count = group.get("count", 0)
                        if name and name != "unknown":
                            top_venues.append({"name": name, "count": count})

        except Exception as exc:
            log.warning("research_trends fetch failed: %s", exc)
            return ToolResult(
                output={
                    "topic": params.topic,
                    "yearly_counts": [],
                    "top_venues": [],
                    "trend": "unknown",
                    "peak_year": None,
                    "total_recent": 0,
                },
                summary=f"Research trends unavailable: {exc}",
            )

        if not yearly_counts:
            return ToolResult(
                output={
                    "topic": params.topic,
                    "yearly_counts": [],
                    "top_venues": [],
                    "trend": "unknown",
                    "peak_year": None,
                    "total_recent": 0,
                },
                summary=f"No trend data found for: {params.topic}",
            )

        trend, peak_year = _compute_trend(yearly_counts)
        total_recent = sum(y["count"] for y in yearly_counts if y["year"] >= current_year - 3)

        await ctx.emit_progress(100, f"Trends ready: {trend} ({len(yearly_counts)} years)")

        return ToolResult(
            output={
                "topic": params.topic,
                "yearly_counts": yearly_counts,
                "top_venues": top_venues[:6],
                "trend": trend,
                "peak_year": peak_year,
                "total_recent": total_recent,
            },
            summary=(
                f"Research trends for '{params.topic}': {trend} "
                f"(peak: {peak_year}, recent 3yr total: {total_recent:,})"
            ),
        )


def _compute_trend(yearly_counts: list[dict]) -> tuple[str, int | None]:
    if len(yearly_counts) < 3:
        return "unknown", None

    counts = [y["count"] for y in yearly_counts]
    years = [y["year"] for y in yearly_counts]
    peak_idx = counts.index(max(counts))
    peak_year = years[peak_idx]

    # Compare last 3 years vs the 3 before that
    if len(counts) >= 6:
        recent = sum(counts[-3:])
        prior = sum(counts[-6:-3])
        if prior == 0:
            trend = "growing"
        elif recent / prior > 1.25:
            trend = "growing"
        elif recent / prior < 0.75:
            trend = "declining"
        else:
            trend = "stable"
    else:
        last = counts[-1]
        first = counts[0]
        trend = "growing" if last > first * 1.2 else "declining" if last < first * 0.8 else "stable"

    return trend, peak_year


research_trends_tool = ResearchTrendsTool()
