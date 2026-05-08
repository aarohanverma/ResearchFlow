"""LLM tool definitions and executors for tool-use (function-calling) workflows.

Tools are provider-agnostic JSON schemas passable to any LLM adapter that supports
function calling (OpenAI, Anthropic, etc.). Each tool has a definition dict (schema
passed to the LLM) and an executor (the Python callable that runs the tool).

Tools: ``SEARCH_TOOL`` (in-app arXiv paper search), ``WEB_SEARCH_TOOL`` (web search
via DuckDuckGo or Tavily), ``ALL_TOOLS`` (list of both).

Usage example (in a workflow node)::

    from app.tools import ALL_TOOLS, ToolExecutor
    from app.adapters.llm import get_llm_adapter

    llm = get_llm_adapter()
    executor = ToolExecutor(db=db)
    result = await llm.complete_with_tools(
        messages=messages,
        model=llm.quality_model,
        tools=ALL_TOOLS,
        tool_executor=executor.execute,
        max_tool_rounds=3,
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# ── Tool definitions (provider-agnostic OpenAI function-calling format) ────────

SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": (
            "Search the indexed arXiv paper database for papers relevant to a query. "
            "Use this when you need more evidence, background, or related work beyond "
            "the provided context. Returns paper titles, TLDRs, key concepts, and "
            "methods sorted by relevance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query: keywords, method names, authors, or a research question. "
                        "Be specific — e.g. 'LoRA parameter-efficient fine-tuning language models' "
                        "rather than just 'fine-tuning'."
                    ),
                },
                "namespace_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of arXiv namespace keys to scope the search "
                        "(e.g. ['cs.AI', 'cs.LG', 'cs.CL']). Omit to search all indexed papers."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1–10, default 5).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

WEB_SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for recent research, documentation, or general information. "
            "Use this for papers, concepts, or events that may not yet be in the local "
            "indexed database, or to verify facts against external sources. "
            "Returns page titles, URLs, and text snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Web search query. Be specific — include method names, paper titles, "
                        "or author names where relevant."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1–10, default 5).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS: list[dict] = [SEARCH_TOOL, WEB_SEARCH_TOOL]


# ── Tool executor ──────────────────────────────────────────────────────────────

class ToolExecutor:
    """Executes LLM tool calls, returning results as JSON-formatted strings.

    Requires an active ``AsyncSession`` for the search tool.  All tool
    results are returned as compact JSON strings suitable for inserting into
    the LLM's tool-result message.

    Args:
        db: Active ``AsyncSession``.  Used by ``search_papers`` to query the
            search repository and embedding adapter.
    """

    def __init__(self, db: Any) -> None:
        """Store a reference to the active database session for tool execution."""
        self._db = db

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call by name and return the result as a JSON string.

        Args:
            tool_name: The function name as returned by the LLM (e.g.
                ``"search_papers"``).
            arguments: Parsed dict of arguments (already JSON-decoded).

        Returns:
            A JSON-encoded string to be placed in the tool-result message.
            Always returns a string — never raises; errors are returned as
            ``{"error": "..."}`` so the LLM can gracefully handle failures.
        """
        if tool_name == "search_papers":
            return await self._search_papers(
                query=arguments.get("query", ""),
                namespace_keys=arguments.get("namespace_keys"),
                limit=min(10, max(1, int(arguments.get("limit", 5)))),
            )
        if tool_name == "web_search":
            return await self._web_search(
                query=arguments.get("query", ""),
                max_results=min(10, max(1, int(arguments.get("max_results", 5)))),
            )
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    async def _web_search(
        self,
        query: str,
        max_results: int,
    ) -> str:
        """Execute a web search and return compact JSON results.

        Uses the configured ``WebSearchAdapter`` (DuckDuckGo by default, or
        Tavily when ``TAVILY_API_KEY`` is configured).

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            JSON string: list of ``{title, url, snippet}`` objects, or
            ``{"error": "..."}`` on failure.
        """
        if not query.strip():
            return json.dumps({"error": "Empty query"})
        try:
            from app.adapters.web_search import get_web_search_adapter
            adapter = get_web_search_adapter()
            results = await adapter.search(query, max_results=max_results)
            return json.dumps({
                "results": [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in results
                ],
                "count": len(results),
                "provider": adapter.provider_id,
            })
        except Exception as exc:
            log.warning("web_search tool failed: %s", exc)
            return json.dumps({"error": str(exc), "results": []})

    async def _search_papers(
        self,
        query: str,
        namespace_keys: list[str] | None,
        limit: int,
    ) -> str:
        """Execute a hybrid paper search and return compact JSON results.

        Embeds the query with ``RETRIEVAL_QUERY`` task type, runs the hybrid
        search repository, and formats the top results as a JSON array.
        Falls back to keyword-only if embedding fails.

        Args:
            query: Search query string.
            namespace_keys: Optional namespace scope.
            limit: Maximum results.

        Returns:
            JSON string: list of ``{title, tldr, namespace_key, key_concepts,
            methods_used, source_url, relevance_score}`` objects, or an
            ``{"error": "..."}`` object on failure.
        """
        if not query.strip():
            return json.dumps({"error": "Empty query"})

        try:
            from app.adapters.embedding import get_embedding_adapter
            from app.repositories.search import SearchRepository

            search_repo = SearchRepository(self._db)
            qvec = None
            embedding_dim = 768
            provider_id = "gemini"

            try:
                embed = get_embedding_adapter()
                qvec = await embed.embed_query(query)
                embedding_dim = embed.dimensions
                provider_id = embed.provider_id
            except Exception as exc:
                log.warning("search_tool: embed_query failed (%s) — keyword-only", exc)

            results = await search_repo.hybrid_search(
                query,
                namespace_keys=namespace_keys,
                query_vector=qvec,
                embedding_dim=embedding_dim,
                embedding_provider=provider_id,
                limit=limit,
            )

            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "tldr": r.get("tldr") or (r.get("abstract", "") or "")[:150],
                    "namespace_key": r.get("namespace_key", ""),
                    "key_concepts": (r.get("key_concepts") or [])[:5],
                    "methods_used": (r.get("methods_used") or [])[:5],
                    "source_url": r.get("source_url", ""),
                    "relevance_score": round(r.get("search_score", 0.0), 3),
                })

            return json.dumps({"papers": formatted, "count": len(formatted)})

        except Exception as exc:
            log.warning("search_tool: search failed: %s", exc)
            return json.dumps({"error": str(exc), "papers": []})
