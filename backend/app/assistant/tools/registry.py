"""In-process registry of assistant tools.

The orchestrator looks up tools by name; the planner sees only the
``describe_for_planner`` view (name + summary + JSON schema) and never
touches implementations.
"""

from __future__ import annotations

from app.assistant.tools.base import AssistantTool

_TOOLS: dict[str, AssistantTool] = {}


def register_tool(tool: AssistantTool) -> None:
    """Register a tool by ``tool.name``. Re-registration overwrites silently
    so module reimports during tests stay idempotent."""
    _TOOLS[tool.name] = tool


def get_tool(name: str) -> AssistantTool | None:
    """Return the tool registered under ``name``, or ``None``."""
    return _TOOLS.get(name)


def list_tools() -> list[AssistantTool]:
    """Return all registered tools, ordered by name for stable display."""
    return [_TOOLS[k] for k in sorted(_TOOLS)]


def describe_for_planner(namespace_key: str | None = None) -> list[dict]:
    """Schema-only view used to brief the planner.

    When ``namespace_key`` is provided, only tools visible for that namespace
    are included (GLOBAL_TOOLS ∪ namespace pack). When None, all tools are
    returned (used by the /tools introspection endpoint).

    Never includes implementation details — only what the planner needs to
    pick a tool and produce valid params.
    """
    if namespace_key is not None:
        from app.assistant.tools.namespace_packs import get_visible_tools
        visible = get_visible_tools(namespace_key)
    else:
        visible = None  # all tools

    out = []
    for tool in list_tools():
        if visible is not None and tool.name not in visible:
            continue
        out.append({
            "name": tool.name,
            "summary": tool.summary,
            "cost_class": tool.cost_class,
            "side_effects": tool.side_effects,
            "cancellable": tool.cancellable,
            "streamable": tool.streamable,
            "input_schema": tool.input_schema.model_json_schema(),
            "output_schema": tool.output_schema.model_json_schema(),
        })
    return out


def reset_registry_for_tests() -> None:
    """Test helper — empty the registry between tests when needed."""
    _TOOLS.clear()
