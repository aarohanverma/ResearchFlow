"""FRED economic data tool — Federal Reserve Economic Data via St. Louis Fed API.

https://fred.stlouisfed.org/docs/api/fred/

Requires FRED_API_KEY in environment (free registration at fred.stlouisfed.org).
Without a key, returns a clear error message instead of failing silently.

Essential for econ.*, q-fin.* namespaces: macroeconomics, monetary policy,
GDP, inflation, unemployment, interest rates, financial indicators.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred"
_TIMEOUT = 15.0


class FredInput(BaseModel):
    query: str = Field(
        min_length=2, max_length=300,
        description=(
            "Search query for economic data series — e.g. 'GDP', 'CPI inflation', "
            "'federal funds rate', 'unemployment rate', 'M2 money supply'. "
            "Also accepts a FRED series ID directly (e.g. 'GDPC1', 'CPIAUCSL', 'FEDFUNDS')."
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)
    fetch_observations: bool = Field(
        default=True,
        description="Whether to also fetch recent data points for the top series.",
    )
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class FredOutput(BaseModel):
    series: list[dict]
    total_count: int


class FredTool:
    """Search FRED for economic time-series data."""

    name = "fred"
    summary = (
        "Search the St. Louis Fed FRED database for economic and financial time-series data. "
        "Use for: GDP, inflation (CPI/PCE), unemployment, interest rates, money supply, "
        "trade balance, housing starts, consumer confidence, exchange rates, and other macro indicators. "
        "Accepts natural-language queries ('US inflation rate') or direct FRED series IDs ('CPIAUCSL'). "
        "Returns series name, description, units, frequency, and recent data points. "
        "Requires FRED_API_KEY env var (free key from fred.stlouisfed.org). "
        "Preferred for econ.*, q-fin.* namespaces."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = FredInput
    output_schema = FredOutput

    def _get_api_key(self) -> str:
        try:
            from app.core.config import get_settings
            return getattr(get_settings(), "fred_api_key", "") or ""
        except Exception:
            return ""

    async def run(self, ctx: ToolContext, params: FredInput) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                output={"series": [], "total_count": 0},
                summary=(
                    "FRED API key not configured. Set FRED_API_KEY in environment "
                    "(free key from https://fred.stlouisfed.org/docs/api/api_key.html)."
                ),
            )

        await ctx.emit_progress(20, f"Searching FRED: {params.query[:60]}")

        q = params.query.strip()
        is_series_id = q.upper() == q and " " not in q and len(q) <= 20

        series_list: list[dict] = []
        total = 0

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                if is_series_id:
                    resp = await client.get(
                        f"{_BASE}/series",
                        params={"series_id": q.upper(), "api_key": api_key, "file_type": "json"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        raw = data.get("seriess") or []
                        total = len(raw)
                        series_list = raw[:params.limit]
                    else:
                        is_series_id = False

                if not is_series_id or not series_list:
                    resp = await client.get(
                        f"{_BASE}/series/search",
                        params={
                            "search_text": q,
                            "limit": params.limit,
                            "order_by": "popularity",
                            "sort_order": "desc",
                            "api_key": api_key,
                            "file_type": "json",
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        total = data.get("count", 0)
                        series_list = (data.get("seriess") or [])[:params.limit]
                    elif resp.status_code == 400:
                        return ToolResult(
                            output={"series": [], "total_count": 0},
                            summary=f"FRED API error: invalid request for query '{q}'",
                        )
                    else:
                        resp.raise_for_status()

        except Exception as exc:
            log.warning("fred search failed: %s", exc)
            return ToolResult(
                output={"series": [], "total_count": 0},
                summary=f"FRED unavailable: {exc}",
            )

        if not series_list:
            return ToolResult(
                output={"series": [], "total_count": 0},
                summary=f"No FRED series found for: {q}",
            )

        await ctx.emit_progress(60, f"Found {len(series_list)} series, fetching recent observations…")

        results: list[dict] = []
        for s in series_list:
            sid = s.get("id", "")
            entry: dict = {
                "series_id": sid,
                "title": s.get("title", ""),
                "observation_start": s.get("observation_start", ""),
                "observation_end": s.get("observation_end", ""),
                "frequency": s.get("frequency_short", s.get("frequency", "")),
                "units": s.get("units_short", s.get("units", "")),
                "seasonal_adjustment": s.get("seasonal_adjustment_short", ""),
                "popularity": s.get("popularity", 0),
                "notes": (s.get("notes") or "")[:300],
                "url": f"https://fred.stlouisfed.org/series/{sid}",
                "observations": [],
                "source": "fred",
            }
            results.append(entry)

        if params.fetch_observations and results:
            top_ids = [r["series_id"] for r in results[:3] if r["series_id"]]
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    for sid in top_ids:
                        obs_resp = await client.get(
                            f"{_BASE}/series/observations",
                            params={
                                "series_id": sid,
                                "limit": 12,
                                "sort_order": "desc",
                                "api_key": api_key,
                                "file_type": "json",
                            },
                        )
                        if obs_resp.status_code == 200:
                            obs_data = obs_resp.json()
                            raw_obs = obs_data.get("observations") or []
                            cleaned = [
                                {"date": o["date"], "value": o["value"]}
                                for o in raw_obs
                                if o.get("value") not in (".", None)
                            ]
                            for r in results:
                                if r["series_id"] == sid:
                                    r["observations"] = list(reversed(cleaned[:12]))
                                    break
            except Exception as exc:
                log.warning("fred observations fetch failed: %s", exc)

        await ctx.emit_progress(100, f"FRED: {len(results)} series found")

        top = results[0]
        obs_note = ""
        if top.get("observations"):
            latest = top["observations"][-1]
            obs_note = f", latest: {latest['value']} {top.get('units', '')} ({latest['date']})"

        return ToolResult(
            output={"series": results, "total_count": total},
            summary=(
                f"{len(results)} FRED series (total: {total:,}) — "
                f"top: '{top['title'][:60]}' [{top['series_id']}]{obs_note}"
            ),
        )


fred_tool = FredTool()
