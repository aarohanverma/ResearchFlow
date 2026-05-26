"""Runtime context wrapper — typed dependency-injection layer for tools.

LangGraph distinguishes four context layers: runtime (run-scoped
dependencies), state (mutable per-run data), store (cross-run memory),
and prompt (curated text shown to the LLM). The user explicitly asked
us to adopt that discipline. We already have most of it — ``ToolContext``
carries the right dependencies, ``LoopState`` carries the per-run
mutable data, the memory subsystem is the store, and the planner /
synthesizer prompts curate what reaches the LLM.

What this module adds:

  * A read-only typed wrapper around ``ToolContext`` so tools can
    access ``runtime.user_id`` / ``runtime.permissions`` etc. with
    autocomplete and refactor-safety, rather than reaching into a
    raw dataclass.

  * Permission scopes derived from the user row (``is_admin``,
    ``feature_overrides``). Tools that need to gate side effects can
    call ``runtime.is_allowed("memory:write")`` instead of duplicating
    the permission lookup. The check still runs against the DB row
    so a revoked permission takes effect immediately.

  * Observability fields (``request_id`` = ``job_id``, ``trace_id``)
    so logs and audit rows from tool code can attribute themselves
    to the originating turn.

  * A ``for_test()`` helper that builds a synthetic runtime so unit
    tests can construct a context without standing up a DB session.

What this module deliberately does NOT do:

  * Replace ``ToolContext``. Every existing tool keeps working
    unchanged — ``RuntimeContext`` is built FROM a ``ToolContext`` and
    exposes the same data via typed properties. New tools can use
    either; we don't force a migration. This is the
    non-disruptive way to adopt the runtime-context pattern.

  * Carry mutable state. Anything that changes mid-run lives on
    LoopState; runtime context is decided at start-of-run and
    frozen for the rest of the turn.

  * Stuff prompt content. The runtime carries IDs and dependencies;
    what reaches the LLM is curated separately by the planner /
    synthesizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.assistant.tools.base import ToolContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeContext:
    """Typed, immutable view of per-turn dependencies.

    All fields come from a ``ToolContext`` or from the User row the
    orchestrator already loaded. Frozen so a tool cannot accidentally
    mutate identity / scope mid-call — if you need mutable state,
    put it on ``LoopState``.
    """

    # ── Identity (who) ───────────────────────────────────────────────
    user_id: UUID
    session_id: UUID

    # ── Scope (where) ────────────────────────────────────────────────
    # ``namespace_key`` is the active namespace; ``namespace_keys`` is
    # the list the user is currently subscribed to (e.g. cs.AI +
    # cs.NLP). Tools that span namespaces use the list; tools that
    # scope to a single bucket use the singular.
    namespace_key: str
    namespace_keys: tuple[str, ...]

    # ── User metadata (how to talk to them) ──────────────────────────
    orientation: str
    expertise_level: str

    # ── Observability (what to attribute) ───────────────────────────
    # ``request_id`` mirrors ``job_id`` — kept distinct so future
    # multi-job runs can supply a separate request id without
    # renaming. ``parent_message_id`` is the assistant turn this
    # tool call is part of.
    request_id: str
    job_id: str
    parent_message_id: UUID | None
    parent_step_id: UUID | None

    # ── Permissions (what's allowed) ────────────────────────────────
    # Soft set of scope strings the calling user holds. Built from
    # the User row's ``is_admin`` flag plus any ``feature_overrides``.
    # Tools check via :meth:`is_allowed`. The check is advisory in
    # this layer — load-bearing auth still lives at the API
    # boundary; this is dependency-injected guidance for tool code.
    permissions: frozenset[str] = field(default_factory=frozenset)

    # ── Feature flags (what's enabled) ──────────────────────────────
    feature_flags: tuple[str, ...] = ()

    # ── Underlying ToolContext ───────────────────────────────────────
    # Carried so tools needing the DB session / progress emitter /
    # cancel-check can still reach them. The "typed" path is the
    # properties above; this is the escape hatch.
    _tool_ctx: ToolContext | None = None

    # ── Permission API ─────────────────────────────────────────────

    def is_allowed(self, scope: str) -> bool:
        """Return True when this runtime holds the given scope.

        Scope strings follow ``namespace:action`` convention
        (e.g. ``memory:write``, ``genie:synthesize``). The check is
        case-sensitive. Wildcard ``*`` in the permissions set means
        "everything"; useful for admin users.
        """
        if "*" in self.permissions:
            return True
        return scope in self.permissions

    def require(self, scope: str) -> None:
        """Raise ``PermissionError`` when the scope isn't held.

        Use in tools whose side effects must hard-gate on permission
        regardless of what the LLM emits — e.g. an admin-only memory
        bulk-delete tool. The LLM cannot bypass this gate by
        rephrasing the request; the check is at the code boundary.
        """
        if not self.is_allowed(scope):
            raise PermissionError(
                f"runtime context lacks required scope {scope!r}",
            )

    # ── Convenience properties ─────────────────────────────────────

    @property
    def db(self):  # noqa: ANN201 — returning AsyncSession would force the import
        """Active DB session — escape hatch for direct queries."""
        return self._tool_ctx.db if self._tool_ctx else None

    @property
    def should_cancel(self):
        return self._tool_ctx.should_cancel if self._tool_ctx else None

    @property
    def emit_progress(self):
        return self._tool_ctx.emit_progress if self._tool_ctx else None

    # ── Construction ───────────────────────────────────────────────

    @classmethod
    def from_tool_context(
        cls,
        ctx: ToolContext,
        *,
        permissions: frozenset[str] = frozenset(),
        feature_flags: tuple[str, ...] = (),
    ) -> "RuntimeContext":
        """Build a runtime view from an existing ``ToolContext``.

        ``permissions`` and ``feature_flags`` are passed in by the
        orchestrator after it loads the User row. They default to
        empty so this builder is safe to call from any code path —
        unauthorized tool code simply gets a context with no
        scopes and ``is_allowed`` returns False for everything.
        """
        return cls(
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            namespace_key=ctx.namespace_key or "",
            namespace_keys=tuple(ctx.namespace_keys or []),
            orientation=ctx.orientation or "",
            expertise_level=ctx.expertise_level or "",
            request_id=ctx.job_id or "",
            job_id=ctx.job_id or "",
            parent_message_id=ctx.parent_message_id,
            parent_step_id=ctx.parent_step_id,
            permissions=permissions,
            feature_flags=feature_flags,
            _tool_ctx=ctx,
        )

    @classmethod
    def for_test(
        cls,
        *,
        user_id: UUID | None = None,
        session_id: UUID | None = None,
        namespace_key: str = "test.namespace",
        permissions: frozenset[str] = frozenset(),
    ) -> "RuntimeContext":
        """Build a synthetic runtime for unit tests.

        Returns a context with no real DB session and no cancel /
        progress callbacks. Tests that exercise tool code should
        either patch the DB layer or construct via the orchestrator
        instead.
        """
        import uuid as _uuid
        return cls(
            user_id=user_id or _uuid.uuid4(),
            session_id=session_id or _uuid.uuid4(),
            namespace_key=namespace_key,
            namespace_keys=(namespace_key,),
            orientation="both",
            expertise_level="practitioner",
            request_id="test-req",
            job_id="test-job",
            parent_message_id=None,
            parent_step_id=None,
            permissions=permissions,
            feature_flags=(),
            _tool_ctx=None,
        )


async def build_runtime_for_user(
    db,
    *,
    ctx: ToolContext,
) -> RuntimeContext:
    """Construct a ``RuntimeContext`` and populate permissions from the
    user row.

    Two permission sources:

      * ``User.is_admin = True`` → grants ``*`` (everything).
      * ``User.feature_overrides`` → JSONB map; each key whose value
        is truthy is added as a scope. Lets admins grant fine-grained
        capabilities (e.g. ``{"memory:bulk_delete": true}``) without
        flipping the admin bit.

    Failure modes — all degrade to "no extra scopes":

      * User row missing (race with delete)
      * ``feature_overrides`` malformed
      * DB error

    Args:
        db: Active ``AsyncSession`` — usually the same one bound to
            ``ctx.db``, but a separate session works too.
        ctx: Tool context for the call; used as the seed.
    """
    from sqlalchemy import select
    from app.models.user import User

    perms: set[str] = set()
    flags: list[str] = []
    try:
        row = await db.execute(select(User).where(User.id == ctx.user_id))
        user = row.scalar_one_or_none()
        if user is not None:
            if getattr(user, "is_admin", False):
                perms.add("*")
            overrides = getattr(user, "feature_overrides", None) or {}
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    if isinstance(k, str) and v:
                        perms.add(k)
                        flags.append(k)
    except Exception as exc:  # noqa: BLE001 — auth is at the API boundary; this layer is advisory
        log.debug("runtime permission lookup failed: %s", exc)

    return RuntimeContext.from_tool_context(
        ctx,
        permissions=frozenset(perms),
        feature_flags=tuple(flags),
    )


__all__ = ["RuntimeContext", "build_runtime_for_user"]
