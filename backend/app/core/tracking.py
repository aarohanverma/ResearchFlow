"""Per-request context for token-usage attribution.

Holds the current authenticated user's UUID and the active workflow / node
labels so that every LLM call made during the request can be recorded against
them automatically. The values are stored in :mod:`contextvars` so they are
isolated between concurrent async requests.

Set by:
    * :func:`app.core.deps.get_current_user_id` — sets ``current_user_id`` on
      every authenticated HTTP request.
    * Workflow nodes — call :func:`set_workflow_context` at entry to attribute
      LLM calls to a specific workflow stage (e.g. ``("study", "assemble")``).

Read by:
    * :class:`app.adapters.llm.tracking.TrackingLLMAdapter` — copies these
      values onto every recorded ``TokenUsage`` row after each LLM completion.
"""

from contextvars import ContextVar
from uuid import UUID

current_user_id: ContextVar[UUID | None] = ContextVar("current_user_id", default=None)
current_workflow: ContextVar[str] = ContextVar("current_workflow", default="")
current_node: ContextVar[str] = ContextVar("current_node", default="")


def set_workflow_context(workflow: str, node: str = "") -> None:
    """Set the current workflow/node labels for token-usage attribution.

    Call this at the start of a workflow node to tag any LLM calls made
    inside that node with the appropriate workflow and node names.

    Args:
        workflow: The workflow identifier (e.g. ``"study"``, ``"genie"``).
        node: Optional node name within the workflow (e.g. ``"hypothesize"``).
    """
    current_workflow.set(workflow)
    current_node.set(node)
