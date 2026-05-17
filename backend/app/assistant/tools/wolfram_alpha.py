"""Wolfram Alpha tool — computational knowledge engine via MCP or direct API.

Uses the official Wolfram Alpha MCP server (hub.docker.com/mcp/server/wolfram-alpha)
via stdio transport when configured, with a direct Short Answers API fallback.

Configuration (any one is sufficient):
  WOLFRAM_MCP_COMMAND   docker run -i --rm -e WOLFRAM_ALPHA_APP_ID=<key> mcp/wolfram-alpha
  WOLFRAM_ALPHA_APP_ID  plain API key for direct HTTP fallback

Capabilities:
  - Mathematics: integrals, ODEs, matrices, statistics, proofs
  - Physics: constants, unit conversions, formulas
  - Chemistry: molecular data, reactions, properties
  - Data lookups: population, geography, historical facts
  - Step-by-step solutions via detailed=True
"""

from __future__ import annotations

import logging
import shlex

from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult
from app.core.config import get_settings

log = logging.getLogger(__name__)

_SHORT_API = "https://api.wolframalpha.com/v1/result"
_FULL_API = "https://api.wolframalpha.com/v2/query"


class WolframAlphaInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000, description="Mathematical or factual query to compute.")
    detailed: bool = Field(
        default=False,
        description=(
            "Set True to return multiple result pods (step-by-step solutions, "
            "alternate forms, numerical approximations). Use for derivations and "
            "proofs where intermediate steps matter."
        ),
    )


class WolframAlphaOutput(BaseModel):
    answer: str
    pods: list[dict]
    query: str
    provider: str


class WolframAlphaTool:
    """Computational knowledge engine — mathematics, science, data lookups."""

    name = "wolfram_alpha"
    summary = (
        "Query Wolfram Alpha's computational knowledge engine for precise, verifiable "
        "answers. Handles mathematics (integrals, equations, statistics, linear algebra, "
        "proofs), physics (constants, unit conversions, formulas), chemistry (properties, "
        "reactions), and factual data (population, distances, historical records). "
        "Set detailed=True for step-by-step solutions. Use whenever the user needs "
        "exact computation that a language model could approximate incorrectly."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = WolframAlphaInput
    output_schema = WolframAlphaOutput

    async def _resolve_app_id(self, ctx: ToolContext) -> str:
        """Return the best available Wolfram App ID: user override > env var."""
        settings = get_settings()
        # Check user-stored key first
        try:
            from app.repositories.user import UserRepository
            repo = UserRepository(ctx.db)
            ps = await repo.get_provider_settings(ctx.user_id)
            if ps and ps.encrypted_wolfram_key:
                return ps.encrypted_wolfram_key
        except Exception:
            pass
        return getattr(settings, "wolfram_alpha_app_id", "") or ""

    async def run(self, ctx: ToolContext, params: WolframAlphaInput) -> ToolResult:
        settings = get_settings()
        mcp_command = getattr(settings, "wolfram_mcp_command", "")
        app_id = await self._resolve_app_id(ctx)

        if not mcp_command and not app_id:
            return ToolResult(
                output={
                    "answer": (
                        "Wolfram Alpha is not available — no API key configured. "
                        "Go to Settings → API Keys and enter your Wolfram Alpha App ID "
                        "(free at developer.wolframalpha.com) to enable this tool."
                    ),
                    "pods": [],
                    "query": params.query,
                    "provider": "wolfram_alpha",
                },
                summary="Wolfram Alpha unavailable — API key not set",
            )

        await ctx.emit_progress(20, f"Querying Wolfram Alpha: {params.query[:60]}")

        # Try MCP server first (Docker-based official server injects its own key)
        if mcp_command:
            result = await self._query_mcp(mcp_command, params.query, params.detailed)
            if result is not None:
                await ctx.emit_progress(100, "Wolfram Alpha MCP answered")
                return result
            log.warning("wolfram_alpha: MCP query failed, falling back to direct API")

        # Direct API fallback
        if app_id:
            result = await self._query_direct(app_id, params.query, params.detailed)
            await ctx.emit_progress(100, "Wolfram Alpha answered")
            return result

        # Both paths failed
        return ToolResult(
            output={"answer": "Wolfram Alpha is unavailable. Check your API key in Settings.", "pods": [], "query": params.query, "provider": "wolfram_alpha"},
            summary="Wolfram Alpha unavailable",
        )

    # ── MCP transport ─────────────────────────────────────────────────────────

    async def _query_mcp(self, command: str, query: str, detailed: bool) -> ToolResult | None:
        """Call the official Wolfram Alpha MCP Docker server via stdio."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            log.warning("wolfram_alpha: langchain-mcp-adapters not installed; using direct API")
            return None

        parts = shlex.split(command)
        if not parts:
            return None

        server_config = {
            "wolfram": {
                "command": parts[0],
                "args": parts[1:],
                "transport": "stdio",
            }
        }

        try:
            async with MultiServerMCPClient(server_config) as client:
                tools = client.get_tools()
                # The official server exposes: query_wolfram_alpha or similar
                tool = next(
                    (t for t in tools if "wolfram" in t.name.lower() or "query" in t.name.lower()),
                    None,
                )
                if not tool:
                    log.warning("wolfram_alpha MCP: no suitable tool found in %s", [t.name for t in tools])
                    return None

                invoke_params = {"query": query}
                if detailed:
                    invoke_params["include_pods"] = True

                raw = await tool.ainvoke(invoke_params)
        except Exception as exc:
            log.warning("wolfram_alpha MCP query failed: %s", exc)
            return None

        return _parse_mcp_result(raw, query)

    # ── Direct API ────────────────────────────────────────────────────────────

    async def _query_direct(self, app_id: str, query: str, detailed: bool) -> ToolResult:
        """Wolfram Alpha HTTP API — Short Answers or Full Results."""
        if detailed:
            return await self._full_query(app_id, query)
        return await self._short_query(app_id, query)

    async def _short_query(self, app_id: str, query: str) -> ToolResult:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    _SHORT_API,
                    params={"appid": app_id, "i": query, "units": "metric"},
                )
                if resp.status_code == 200:
                    answer = resp.text.strip()
                    return ToolResult(
                        output={"answer": answer, "pods": [], "query": query, "provider": "wolfram_alpha_api"},
                        summary=f"Wolfram Alpha: {answer[:120]}",
                    )
                if resp.status_code == 501:
                    # Not understood — try full API
                    return await self._full_query(app_id, query)
                return ToolResult(
                    output={"answer": f"No result (HTTP {resp.status_code})", "pods": [], "query": query, "provider": "wolfram_alpha_api"},
                    summary=f"Wolfram Alpha returned HTTP {resp.status_code}",
                )
        except Exception as exc:
            log.warning("wolfram_alpha short query failed: %s", exc)
            return ToolResult(
                output={"answer": f"Query failed: {exc}", "pods": [], "query": query, "provider": "wolfram_alpha_api"},
                summary=f"Wolfram Alpha error: {type(exc).__name__}",
            )

    async def _full_query(self, app_id: str, query: str) -> ToolResult:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    _FULL_API,
                    params={"appid": app_id, "input": query, "format": "plaintext", "output": "json", "units": "metric"},
                )
                if resp.status_code != 200:
                    return ToolResult(
                        output={"answer": f"No result (HTTP {resp.status_code})", "pods": [], "query": query, "provider": "wolfram_alpha_api"},
                        summary=f"Wolfram Alpha returned HTTP {resp.status_code}",
                    )
                data = resp.json()
        except Exception as exc:
            log.warning("wolfram_alpha full query failed: %s", exc)
            return ToolResult(
                output={"answer": f"Query failed: {exc}", "pods": [], "query": query, "provider": "wolfram_alpha_api"},
                summary=f"Wolfram Alpha error: {type(exc).__name__}",
            )

        qr = data.get("queryresult", {})
        if not qr.get("success"):
            suggestions = qr.get("didyoumean", [])
            hint = ""
            if suggestions:
                vals = [d.get("val", "") for d in (suggestions if isinstance(suggestions, list) else [suggestions])]
                hint = f" Did you mean: {', '.join(vals[:3])}?"
            return ToolResult(
                output={"answer": f"Wolfram Alpha couldn't interpret the query.{hint}", "pods": [], "query": query, "provider": "wolfram_alpha_api"},
                summary="Wolfram Alpha: query not understood",
            )

        pods: list[dict] = []
        answer_parts: list[str] = []
        priority_titles = {"result", "solution", "value", "decimal form", "exact result", "input interpretation"}
        for pod in qr.get("pods", []):
            title = pod.get("title", "")
            texts = [
                sp.get("plaintext", "").strip()
                for sp in pod.get("subpods", [])
                if sp.get("plaintext", "").strip()
            ]
            if texts:
                pods.append({"title": title, "text": "\n".join(texts)})
                if title.lower() in priority_titles:
                    answer_parts.append(f"**{title}**: {'; '.join(texts)}")

        if not answer_parts and pods:
            for p in pods[:2]:
                answer_parts.append(f"**{p['title']}**: {p['text'][:200]}")

        answer = "\n".join(answer_parts) or "No interpretable result returned."
        return ToolResult(
            output={"answer": answer, "pods": pods[:10], "query": query, "provider": "wolfram_alpha_api"},
            summary=f"Wolfram Alpha: {answer[:120]}",
        )


def _parse_mcp_result(raw: object, query: str) -> ToolResult:
    """Normalise whatever the MCP tool returns into our ToolResult shape."""
    if isinstance(raw, str):
        answer = raw.strip()
    elif isinstance(raw, dict):
        answer = raw.get("result") or raw.get("answer") or raw.get("text") or str(raw)
    elif isinstance(raw, list) and raw:
        answer = "\n".join(str(item) for item in raw if item)
    else:
        answer = str(raw)

    return ToolResult(
        output={"answer": answer, "pods": [], "query": query, "provider": "wolfram_alpha_mcp"},
        summary=f"Wolfram Alpha: {answer[:120]}",
    )


wolfram_alpha_tool = WolframAlphaTool()
