"""arXiv MCP source — advanced ingestion via Model Context Protocol.

Configured via:
  ARXIV_MCP_TRANSPORT=stdio | sse
  ARXIV_MCP_COMMAND=python -m arxiv_mcp_server --storage-path /data/papers   (stdio)
  ARXIV_MCP_URL=http://localhost:8765/sse                                    (sse)

SECURITY: paper text is treated as DATA only — never as instructions.
"""

import logging
import re
import shlex
from datetime import datetime, timezone

import feedparser
import httpx

from app.adapters.sources.base import BaseSource, RawPaper
from app.core.config import settings

log = logging.getLogger(__name__)


def _stdio_server_config() -> dict:
    """Build a langchain-mcp-adapters StdioConnection from ARXIV_MCP_COMMAND.

    The setting is a shell-style string ("python -m arxiv_mcp_server …"); the
    adapter needs the executable and its args split out.
    """
    parts = shlex.split(settings.arxiv_mcp_command)
    if not parts:
        raise ValueError("ARXIV_MCP_COMMAND is empty")
    return {
        "arxiv": {
            "command": parts[0],
            "args": parts[1:],
            "transport": "stdio",
        }
    }


def _sse_server_config() -> dict:
    return {
        "arxiv": {
            "url": settings.arxiv_mcp_url,
            "transport": "sse",
        }
    }


def _server_config() -> dict:
    if settings.arxiv_mcp_transport == "stdio":
        return _stdio_server_config()
    return _sse_server_config()


class ArXivMcpSource(BaseSource):
    """Ingestion source that retrieves arXiv papers via a Model Context Protocol server.

    Supports both ``stdio`` and ``sse`` transports configured through
    ``settings.arxiv_mcp_transport``. Falls back to ``ArXivRssSource`` if
    the ``langchain-mcp-adapters`` package is not installed.
    """

    source_name = "arxiv_mcp"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        namespace_keys: list[str] | None = None,
    ) -> list[RawPaper]:
        """Search arXiv through MCP, falling back to the public arXiv Atom API.

        Args:
            query: Natural-language search query.
            max_results: Maximum number of papers to return.
            namespace_keys: Optional arXiv categories used to constrain the
                search and assign imported papers to the active ResearchFlow
                namespace.

        Returns:
            Normalized ``RawPaper`` rows suitable for feed import.
        """
        try:
            papers = await self._search_mcp(query, max_results=max_results)
        except Exception as exc:
            log.warning("arxiv_mcp search failed; falling back to Atom API: %s", exc)
            papers = []

        if not papers:
            papers = await self._search_atom_api(
                query, max_results=max_results, namespace_keys=namespace_keys
            )

        allowed = set(namespace_keys or [])
        if allowed:
            for p in papers:
                if p.namespace_key not in allowed:
                    # Keep imported papers in the active namespace when arXiv
                    # returns a primary category outside the user's current scope.
                    p.namespace_key = namespace_keys[0]
        return papers[:max_results]

    async def fetch(self, external_category_key: str) -> list[RawPaper]:
        """Connect to arXiv MCP server and call search_papers tool."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            log.warning("langchain-mcp-adapters not installed; falling back to RSS")
            from app.adapters.sources.arxiv_rss import ArXivRssSource
            return await ArXivRssSource().fetch(external_category_key)

        async with MultiServerMCPClient(_server_config()) as client:
            tools = client.get_tools()
            search_tool = next((t for t in tools if t.name == "search_papers"), None)
            if not search_tool:
                log.error("MCP server missing search_papers tool")
                return []

            # SECURITY: invoke tool with a safe, bounded query — not user-supplied text
            result = await search_tool.ainvoke(
                {"query": external_category_key, "max_results": 50}
            )

        papers: list[RawPaper] = []
        for item in result if isinstance(result, list) else []:
            arxiv_id = item.get("arxiv_id", "")
            if not arxiv_id:
                continue
            papers.append(RawPaper(
                external_id=arxiv_id,
                title=item.get("title", ""),
                authors=item.get("authors", []),
                abstract=item.get("abstract", ""),
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=None,
                namespace_key=external_category_key,
                raw=item,
            ))
        return papers

    async def _search_mcp(self, query: str, *, max_results: int) -> list[RawPaper]:
        """Call the MCP ``search_papers`` tool with a user search query."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            return []

        async with MultiServerMCPClient(_server_config()) as client:
            tools = client.get_tools()
            search_tool = next((t for t in tools if t.name == "search_papers"), None)
            if not search_tool:
                return []
            result = await search_tool.ainvoke({"query": query, "max_results": max_results})

        papers: list[RawPaper] = []
        for item in result if isinstance(result, list) else []:
            arxiv_id = item.get("arxiv_id") or item.get("id") or ""
            arxiv_id = _extract_arxiv_id(str(arxiv_id)) or str(arxiv_id).strip()
            if not arxiv_id:
                continue
            namespace_key = (
                item.get("primary_category")
                or item.get("category")
                or (item.get("categories") or ["cs.AI"])[0]
            )
            papers.append(RawPaper(
                external_id=arxiv_id,
                title=item.get("title", ""),
                authors=item.get("authors", []),
                abstract=item.get("abstract", item.get("summary", "")),
                source_url=item.get("source_url") or f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=item.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=_parse_maybe_date(item.get("published") or item.get("published_at")),
                namespace_key=str(namespace_key),
                raw=item,
            ))
        return papers

    async def _search_atom_api(
        self,
        query: str,
        *,
        max_results: int,
        namespace_keys: list[str] | None,
    ) -> list[RawPaper]:
        """Fallback arXiv Atom API search used when MCP is unavailable."""
        cat_filter = ""
        if namespace_keys:
            cat_filter = " OR ".join(f"cat:{ns}" for ns in namespace_keys)
        search_query = f"all:{query}"
        if cat_filter:
            search_query = f"({search_query}) AND ({cat_filter})"

        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": "0",
            "max_results": str(max_results),
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get("https://export.arxiv.org/api/query", params=params)
                resp.raise_for_status()
        except Exception as exc:
            log.warning("arxiv_mcp Atom fallback failed query=%r err=%s", query, exc)
            return []

        feed = feedparser.parse(resp.text)
        papers: list[RawPaper] = []
        for entry in feed.entries:
            arxiv_id = _extract_arxiv_id(entry.get("id", ""))
            if not arxiv_id:
                continue
            authors = [a.get("name", "Unknown") for a in entry.get("authors", [])] or ["Unknown"]
            tags = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
            namespace_key = tags[0] if tags else (namespace_keys or ["cs.AI"])[0]
            papers.append(RawPaper(
                external_id=arxiv_id,
                title=re.sub(r"\s+", " ", entry.get("title", "")).strip(),
                authors=authors,
                abstract=entry.get("summary", "").strip(),
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=_parse_maybe_date(entry.get("published")),
                namespace_key=namespace_key,
                raw=dict(entry),
            ))
        return papers


def get_source(source_name: str) -> BaseSource:
    """Return a ``BaseSource`` instance for the given source name.

    Args:
        source_name: Source identifier (``"arxiv_mcp"`` or ``"arxiv_rss"``).
            When ``settings.ingestion_mode`` is ``"mcp"``, always returns
            ``ArXivMcpSource`` regardless of ``source_name``.

    Returns:
        An instantiated ``BaseSource`` for the requested source.
    """
    if source_name == "arxiv_mcp" or settings.ingestion_mode == "mcp":
        return ArXivMcpSource()
    return __import__(
        "app.adapters.sources.arxiv_rss", fromlist=["ArXivRssSource"]
    ).ArXivRssSource()


def _extract_arxiv_id(value: str) -> str | None:
    """Extract canonical arXiv ID and strip any version suffix."""
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", value)
    return m.group(1) if m else None


def _parse_maybe_date(value: object) -> datetime | None:
    """Parse common arXiv date strings into timezone-aware UTC datetimes."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
