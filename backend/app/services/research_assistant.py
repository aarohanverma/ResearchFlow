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


async def replay_turn(
    *,
    user_id: UUID,
    session_id: UUID,
    message_id: UUID,
    new_content: str | None = None,
) -> tuple[UUID, UUID, str]:
    """Edit / regenerate a turn — replays from the given message onward.

    Two modes:

    * ``new_content`` is provided AND the message is a user turn:
      treat as an edit. Replace the user message's content with
      ``new_content``, delete every message whose ``created_at`` is
      strictly after it (including the paired assistant reply and
      anything downstream), then queue a fresh orchestrator turn against
      the edited content.

    * ``new_content`` is ``None`` AND the message is an assistant turn:
      treat as a regenerate. Walk back to the most-recent user message
      before the target, drop everything from the assistant message
      onward (including the assistant message itself), then re-queue
      that user message's content as a fresh turn.

    Returns ``(user_message_id, new_assistant_message_id, job_id)``.
    """
    from sqlalchemy import delete as sql_delete, select
    from app.models.assistant import AssistantMessage, AssistantTask

    downstream_job_ids: list[str] = []

    async with async_session_factory() as db:
        repo = AssistantRepository(db)
        session = await repo.get_session(user_id, session_id)
        if not session:
            raise ValueError("session not found")

        target = next(
            (m for m in (session.messages or []) if m.id == message_id),
            None,
        )
        if target is None:
            raise ValueError("message not found")

        target_role = target.role.value if hasattr(target.role, "value") else str(target.role)

        if new_content is not None:
            if target_role != "user":
                raise ValueError("Only user messages can be edited")
            content = new_content.strip()
            if not content:
                raise ValueError("Edited content cannot be empty")
            # Keep the user message — overwrite content, drop everything after.
            cutoff_msg = target
            cutoff_ts = target.created_at
            keep_target = True
        else:
            if target_role != "assistant":
                raise ValueError("Only assistant messages can be regenerated")
            # Find the most recent user message strictly before the assistant.
            prior_user = None
            for m in (session.messages or []):
                m_role = m.role.value if hasattr(m.role, "value") else str(m.role)
                if m_role == "user" and m.created_at < target.created_at:
                    if prior_user is None or m.created_at > prior_user.created_at:
                        prior_user = m
            if prior_user is None:
                raise ValueError("No preceding user message to regenerate from")
            content = prior_user.content or ""
            cutoff_msg = target  # delete the assistant + everything after
            cutoff_ts = target.created_at
            keep_target = False

        # Capture downstream message IDs so we can cancel their tasks too.
        downstream_ids: list[UUID] = []
        for m in (session.messages or []):
            if m.id == cutoff_msg.id:
                if not keep_target:
                    downstream_ids.append(m.id)
                continue
            if m.created_at > cutoff_ts:
                downstream_ids.append(m.id)

        # Cancel any tasks whose assistant message we're about to delete so
        # an in-flight job can't write back to a row that no longer exists.
        if downstream_ids:
            # Snapshot the job ids BEFORE the status update so we can
            # signal the in-process scheduler to actually interrupt the
            # asyncio.Task — flipping the DB row to "cancelled" alone
            # leaves the running coroutine churning until it checks the
            # cancellation gate (and would still race to write the
            # finalised message back to a row we're about to delete).
            job_rows = await db.execute(
                select(AssistantTask.job_id).where(
                    AssistantTask.session_id == session.id,
                    AssistantTask.assistant_message_id.in_(downstream_ids),
                    AssistantTask.status.in_(["pending", "running"]),
                )
            )
            downstream_job_ids = [str(r[0]) for r in job_rows.fetchall() if r[0]]

            await db.execute(
                AssistantTask.__table__.update()
                .where(
                    AssistantTask.session_id == session.id,
                    AssistantTask.assistant_message_id.in_(downstream_ids),
                    AssistantTask.status.in_(["pending", "running"]),
                )
                .values(
                    status="cancelled",
                    cancel_requested_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.execute(
                sql_delete(AssistantMessage).where(
                    AssistantMessage.session_id == session.id,
                    AssistantMessage.id.in_(downstream_ids),
                )
            )

        if keep_target:
            target.content = content
            target.payload = {**(target.payload or {}), "edited_at": datetime.now(timezone.utc).isoformat()}
            user_msg_id = target.id
        else:
            user_msg_id = prior_user.id

        # Insert a fresh assistant placeholder for the new turn.
        new_assistant_msg = await repo.add_message(
            session_id=session.id,
            user_id=user_id,
            role=AssistantMessageRole.assistant,
            content="",
            message_type="workflow",
            payload={"status": "running", "blocks": [], "workflow": {"actions": [], "trace": []}},
        )

        namespace_key = session.namespace_key or "cs.AI"
        job_id = f"assistant:{uuid.uuid4()}"
        await repo.create_task(
            job_id=job_id,
            session_id=session.id,
            user_id=user_id,
            assistant_message_id=new_assistant_msg.id,
            task_type="assistant",
            title=_task_title(content),
            namespace_key=namespace_key,
            progress={"stage": "queued", "percent": 0, "summary": "Queued assistant orchestration"},
        )
        await db.commit()

    # Signal the in-process scheduler to interrupt any downstream
    # asyncio.Tasks whose message rows we just deleted. Without this the
    # orchestrator coroutine keeps running and races to write a finalised
    # message back to a non-existent AssistantMessage row, producing
    # FK errors in the log and (more importantly) wasting LLM spend on
    # work that will be discarded.
    for stale_job_id in downstream_job_ids:
        try:
            scheduler.cancel(stale_job_id)
        except Exception as exc:
            log.debug("replay_turn: scheduler.cancel(%s) failed: %s", stale_job_id, exc)
        # Mirror the cancellation in the job store so the notification panel
        # stops showing the spinner for the orphaned job.
        try:
            await get_job_store().update(stale_job_id, {
                "status": "cancelled",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "summary": "Superseded by replay",
            })
        except Exception:
            pass

    await get_job_store().put(job_id, {
        "kind": "assistant",
        "job_id": job_id,
        "user_id": str(user_id),
        "session_id": str(session_id),
        "assistant_message_id": str(new_assistant_msg.id),
        "title": _task_title(content),
        "status": "running",
        "namespace_key": namespace_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "summary": "Assistant task started (replay)",
    })

    scheduler.submit(job_id)
    return user_msg_id, new_assistant_msg.id, job_id


async def cancel_task(user_id: UUID, job_id: str) -> bool:
    """Cancel an assistant task in the DB and the in-process runner.

    The full cancellation sequence is:

    1. Flip the AssistantTask row to ``cancelled`` and stamp
       ``cancel_requested_at`` so the orchestrator's cooperative
       cancellation gate sees the request on its next check.
    2. Update the paired AssistantMessage placeholder so the UI does not
       render a perpetual "running" bubble when the orchestrator was never
       running in this worker (e.g. cancelled before scheduler picked it
       up, or running on a different worker entirely).
    3. Best-effort interrupt the in-process asyncio.Task — returns False
       silently when the task lives on another worker (cooperative cancel
       still applies because the orchestrator polls the DB flag).
    4. Mirror the new status into the JobStore so the notification panel
       updates immediately without waiting for a poll round-trip.
    """
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
        # Stamp the paired assistant message so the chat UI doesn't show
        # a forever-spinning bubble for cancellations that hit before any
        # progress was published (e.g. user clicks cancel within ms of
        # submit, or the task was queued on a different worker that the
        # cooperative cancel cannot reach in this process).
        if task.assistant_message_id:
            try:
                await repo.update_message(
                    task.assistant_message_id,
                    content="Cancelled. Partial results, if any, were left safely in the workspace.",
                    payload={"status": "cancelled"},
                    message_type="workflow",
                )
            except Exception as exc:
                log.debug("cancel_task: message update failed: %s", exc)
        await db.commit()

    scheduler.cancel(job_id)

    await get_job_store().update(job_id, {
        "status": "cancelled",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "summary": "Assistant task cancelled",
    })

    # Close the event bus channel so any open SSE subscribers exit cleanly
    # rather than waiting on a task_completed/task_failed event that the
    # orchestrator may not get a chance to publish if it was killed
    # before reaching its cooperative cancellation gate.
    try:
        from app.assistant.events import get_event_bus as _bus
        _bus().close(job_id)
    except Exception:
        pass
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
