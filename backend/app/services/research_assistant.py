"""Research Assistant service layer.

Owns session/message/task bookkeeping and JobStore notifications. The actual
turn execution is delegated to ``app.assistant.orchestrator.Orchestrator``,
which writes per-step rows, runs the tool registry, and synthesizes answers.

Tool catalogue: see ``app.assistant.tools.*``. Adding a new capability =
register a new AssistantTool — no changes to this file or the API layer.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from uuid import UUID

# Importing the assistant package registers all built-in tools at startup.
import app.assistant.tools  # noqa: F401
from app.assistant import scheduler
from app.assistant.orchestrator import Orchestrator
from app.db.session import async_session_factory
from app.models.assistant import AssistantMessageRole, AssistantTaskStatus
from app.repositories.assistant import AssistantRepository
from app.repositories.user import UserRepository
from app.services.job_store import get_job_store

log = logging.getLogger(__name__)

_orchestrator = Orchestrator()
# Hand the orchestrator to the scheduler module so submit()/cancel() route
# to its run_turn coroutine. Done here (not in the scheduler) to avoid a
# circular import at module load time.
scheduler.register_runner(_orchestrator.run_turn)


async def create_session(
    *,
    user_id: UUID,
    namespace_key: str,
    topic_keys: list[str],
    title: str | None = None,
) -> UUID:
    """Create a persistent assistant session and return its ID."""
    async with async_session_factory() as db:
        repo = AssistantRepository(db)
        user_repo = UserRepository(db)
        user = await user_repo.get_by_id(user_id)
        session = await repo.create_session(
            user_id=user_id,
            title=title or _default_title(namespace_key, topic_keys),
            namespace_key=namespace_key,
            topic_keys=topic_keys or [namespace_key],
            orientation=user.orientation.value if user else "both",
            expertise_level=user.expertise_level.value if user else "practitioner",
            state={"namespace_key": namespace_key, "topic_keys": topic_keys or [namespace_key]},
        )
        await repo.add_message(
            session_id=session.id,
            user_id=user_id,
            role=AssistantMessageRole.system,
            content=(
                "Research workspace created. I will keep searches, papers, graph state, "
                "and generated outputs attached to this investigation."
            ),
            message_type="system",
            payload={"kind": "workspace_created"},
        )
        await db.commit()
        return session.id


async def branch_session(
    *,
    user_id: UUID,
    source_session_id: UUID,
    from_message_id: UUID | None = None,
    title: str | None = None,
) -> UUID | None:
    """Create a child session that preserves the source session context.

    Max nesting depth is 3 levels (root → L1 → L2 → L3). Attempting to
    branch from an L3 session returns None so the API surfaces a 404.
    """
    async with async_session_factory() as db:
        repo = AssistantRepository(db)
        source = await repo.get_session(user_id, source_session_id)
        if not source:
            return None

        # Walk up the parent chain to count depth (cap at 4 to avoid unbounded queries).
        depth = 0
        node = source
        while node.parent_session_id and depth < 4:
            parent = await repo.get_session(user_id, node.parent_session_id)
            if not parent:
                break
            node = parent
            depth += 1
        if depth >= 3:
            # Already at maximum nesting depth.
            return None
        child = await repo.create_session(
            user_id=user_id,
            title=title or f"Branch: {source.title}",
            namespace_key=source.namespace_key,
            topic_keys=list(source.topic_keys or []),
            orientation=source.orientation,
            expertise_level=source.expertise_level,
            parent_session_id=source.id,
            branch_from_message_id=from_message_id,
            state={
                **(source.state or {}),
                "branched_from": str(source.id),
                "branch_from_message_id": str(from_message_id) if from_message_id else None,
            },
        )
        await repo.add_message(
            session_id=child.id,
            user_id=user_id,
            role=AssistantMessageRole.system,
            content=(
                "Branched from the prior investigation. Prior papers, citations, and "
                "workflow references remain traceable through the parent link."
            ),
            message_type="system",
            payload={"kind": "branch", "parent_session_id": str(source.id)},
        )
        await db.commit()
        return child.id


async def submit_turn(
    *,
    user_id: UUID,
    session_id: UUID,
    content: str,
    namespace_key: str,
    topic_keys: list[str],
    attachments: list[dict] | None = None,
) -> tuple[UUID, UUID, str]:
    """Persist a user turn, queue orchestration, and return IDs.

    Returns:
        ``(user_message_id, assistant_message_id, job_id)``.
    """
    async with async_session_factory() as db:
        repo = AssistantRepository(db)
        session = await repo.get_session(user_id, session_id)
        if not session:
            raise ValueError("session not found")

        session.namespace_key = namespace_key
        session.topic_keys = topic_keys or [namespace_key]
        session.state = {
            **(session.state or {}),
            "namespace_key": namespace_key,
            "topic_keys": topic_keys or [namespace_key],
            "last_user_intent": content[:500],
        }
        # Auto-rename on the first user turn so the session list reflects the
        # actual investigation rather than a generic placeholder. We treat any
        # title that still starts with our default ("Research workspace:") OR
        # "Branch:" as up-for-grabs; user-edited titles are preserved.
        already_has_user_msg = any(
            (m.role.value if hasattr(m.role, "value") else str(m.role)) == "user"
            for m in (session.messages or [])
        )
        if not already_has_user_msg:
            current_title = session.title or ""
            if (
                current_title.startswith("Research workspace:")
                or current_title.startswith("Branch:")
                or current_title == "Untitled investigation"
            ):
                session.title = _title_from_query(content)

        user_msg = await repo.add_message(
            session_id=session.id,
            user_id=user_id,
            role=AssistantMessageRole.user,
            content=content,
            payload={"attachments": attachments or []},
        )
        assistant_msg = await repo.add_message(
            session_id=session.id,
            user_id=user_id,
            role=AssistantMessageRole.assistant,
            content="",
            message_type="workflow",
            payload={"status": "running", "blocks": [], "workflow": {"actions": [], "trace": []}},
        )

        job_id = f"assistant:{uuid.uuid4()}"
        task = await repo.create_task(
            job_id=job_id,
            session_id=session.id,
            user_id=user_id,
            assistant_message_id=assistant_msg.id,
            task_type="assistant",
            title=_task_title(content),
            namespace_key=namespace_key,
            progress={"stage": "queued", "percent": 0, "summary": "Queued assistant orchestration"},
        )
        await db.commit()

    await get_job_store().put(job_id, {
        "kind": "assistant",
        "job_id": job_id,
        "user_id": str(user_id),
        "session_id": str(session_id),
        "assistant_message_id": str(assistant_msg.id),
        "task_id": str(task.id),
        "title": task.title,
        "status": "running",
        "namespace_key": namespace_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "summary": "Assistant task started",
    })

    scheduler.submit(job_id)
    return user_msg.id, assistant_msg.id, job_id


async def cancel_task(user_id: UUID, job_id: str) -> bool:
    """Cancel an assistant task in the DB and the in-process runner."""
    async with async_session_factory() as db:
        repo = AssistantRepository(db)
        task = await repo.get_task_by_job_id(user_id, job_id)
        if not task:
            return False
        await repo.update_task(
            job_id,
            status=AssistantTaskStatus.cancelled,
            progress={"stage": "cancelled", "percent": 100, "summary": "Cancellation requested"},
            completed=True,
            cancel_requested=True,
        )
        await db.commit()

    scheduler.cancel(job_id)

    await get_job_store().update(job_id, {
        "status": "cancelled",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "summary": "Assistant task cancelled",
    })
    return True


def _default_title(namespace_key: str, topic_keys: list[str]) -> str:
    if topic_keys:
        return f"Research workspace: {', '.join(topic_keys[:3])}"
    return f"Research workspace: {namespace_key}"


def _task_title(content: str) -> str:
    clean = " ".join(content.strip().split())
    return clean[:80] or "Assistant workflow"


def _title_from_query(content: str) -> str:
    """Derive a session title from the user's first turn.

    Truncates at the first sentence boundary or 60 chars, whichever comes
    first, and strips trailing question marks for readability. Never empty —
    falls back to a sane placeholder if the message is whitespace.
    """
    import re

    raw = " ".join((content or "").strip().split())
    if not raw:
        return "Untitled investigation"
    # Cut at the first sentence break to keep titles tight.
    m = re.search(r"[.!?]", raw)
    if m and m.start() > 8:
        raw = raw[: m.start()]
    if len(raw) > 60:
        raw = raw[:57].rstrip() + "…"
    # Title-case looks weird for queries; preserve the user's casing.
    return raw or "Untitled investigation"
