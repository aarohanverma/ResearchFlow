"""Base classes for middlewares.

Concrete middlewares inherit from :class:`BaseMiddleware` so they only
override the hooks they care about. The default implementations are
no-ops that return the appropriate "continue normally" sentinels.

``NoopMiddleware`` is exported for tests that want to validate the
chain composition without exercising any real cross-cutting behavior.
"""

from __future__ import annotations

from typing import Any

from app.assistant.react.middleware import (
    CONTINUE,
    FinalizeAllow,
    FinalizeGate,
    PreDispatchResult,
)
from app.assistant.tools.base import ToolResult


class BaseMiddleware:
    """No-op default implementations of every hook.

    Subclass and override only the hooks you need. Keeping the base
    class concrete (rather than abstract) means a partial middleware
    that only handles ``after_tool`` doesn't have to write five empty
    methods to satisfy the Protocol.
    """

    name: str = "base"

    async def before_iteration(self, state: Any) -> None:
        return None

    async def before_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
    ) -> PreDispatchResult:
        return CONTINUE

    async def after_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        return None

    async def on_tool_error(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        exc: BaseException,
    ) -> None:
        return None

    async def gate_finalize(self, state: Any) -> FinalizeGate:
        return FinalizeAllow()


class NoopMiddleware(BaseMiddleware):
    """Pure no-op — useful as a test fixture or as an explicit
    placeholder in a chain that wants to keep a slot for future
    middleware without binding it yet."""

    name = "noop"
