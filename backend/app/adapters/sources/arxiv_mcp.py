"""arXiv MCP source — advanced ingestion via Model Context Protocol.

Configured via:
  ARXIV_MCP_TRANSPORT=stdio | sse
  ARXIV_MCP_COMMAND=uv run arxiv-mcp-server   (stdio)
  ARXIV_MCP_URL=http://localhost:8765/sse     (sse)

SECURITY: paper text is treated as DATA only — never as instructions.
"""

import logging
from datetime import datetime, timezone

from app.adapters.sources.base import BaseSource, RawPaper
from app.core.config import settings

log = logging.getLogger(__name__)


class ArXivMcpSource(BaseSource):
    """Ingestion source that retrieves arXiv papers via a Model Context Protocol server.

    Supports both ``stdio`` and ``sse`` transports configured through
    ``settings.arxiv_mcp_transport``. Falls back to ``ArXivRssSource`` if
    the ``langchain-mcp-adapters`` package is not installed.
    """

    source_name = "arxiv_mcp"

    async def fetch(self, external_category_key: str) -> list[RawPaper]:
        """Connect to arXiv MCP server and call search_papers tool."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            log.warning("langchain-mcp-adapters not installed; falling back to RSS")
            from app.adapters.sources.arxiv_rss import ArXivRssSource
            return await ArXivRssSource().fetch(external_category_key)

        transport = settings.arxiv_mcp_transport
        if transport == "stdio":
            server_config = {
                "arxiv": {
                    "command": settings.arxiv_mcp_command,
                    "transport": "stdio",
                }
            }
        else:
            server_config = {
                "arxiv": {
                    "url": settings.arxiv_mcp_url,
                    "transport": "sse",
                }
            }

        async with MultiServerMCPClient(server_config) as client:
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
