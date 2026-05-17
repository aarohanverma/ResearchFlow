"""Research Assistant orchestration package.

Tool registry, orchestrator, and planner. Wraps existing platform capabilities
(Deep Search, arXiv MCP, Genie, Study, Graph) as composable tools rather than
re-implementing them.
"""

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.assistant.tools.registry import (
    describe_for_planner,
    get_tool,
    list_tools,
    register_tool,
)

__all__ = [
    "AssistantTool",
    "ToolContext",
    "ToolResult",
    "describe_for_planner",
    "get_tool",
    "list_tools",
    "register_tool",
]
