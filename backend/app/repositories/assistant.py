"""Repository for persistent Research Assistant sessions, messages, and tasks."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.assistant import (
    AssistantArtifact,
    AssistantMessage,
    AssistantMessageRole,
    AssistantSession,
    AssistantSessionStatus,
    AssistantStep,
    AssistantStepStatus,
    AssistantTask,
    AssistantTaskStatus,
)


class AssistantRepository:
    """Data-access layer for the Research Assistant workspace."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def list_sessions(
        self,
        user_id: UUID,
        limit: int = 50,
        namespace_key: str | None = None,
    ) -> list[AssistantSession]:
        q = (
            select(AssistantSession)
            .options(
                selectinload(AssistantSession.messages),
                selectinload(AssistantSession.tasks),
            )
            .where(
                AssistantSession.user_id == user_id,
                AssistantSession.status == AssistantSessionStatus.active,
            )
        )
        if namespace_key:
            q = q.where(AssistantSession.namespace_key == namespace_key)
        q = q.order_by(AssistantSession.updated_at.desc()).limit(limit)
        result = await self._db.execute(q)
        return list(result.scalars())

    async def get_session(self, user_id: UUID, session_id: UUID) -> AssistantSession | None:
        result = await self._db.execute(
            select(AssistantSession)
            .options(
                selectinload(AssistantSession.messages),
                selectinload(AssistantSession.tasks),
            )
            .where(
                AssistantSession.user_id == user_id,
                AssistantSession.id == session_id,
            )
        )
        return result.scalar_one_or_none()

    async def create_session(
        self,
        *,
        user_id: UUID,
        title: str,
        namespace_key: str,
        topic_keys: list[str],
        orientation: str,
        expertise_level: str,
        parent_session_id: UUID | None = None,
        branch_from_message_id: UUID | None = None,
        state: dict | None = None,
    ) -> AssistantSession:
        # Inherit namespace-level long-term memory from the most recent session
        # in the same namespace so insights persist across session boundaries.
        initial_state = dict(state or {})
        if not parent_session_id and not initial_state.get("ns_memory"):
            try:
                recent = await self._db.execute(
                    select(AssistantSession)
                    .where(
                        AssistantSession.user_id == user_id,
                        AssistantSession.namespace_key == namespace_key,
                        AssistantSession.status == AssistantSessionStatus.active,
                    )
                    .order_by(AssistantSession.updated_at.desc())
                    .limit(1)
                )
                prev = recent.scalar_one_or_none()
                if prev and prev.state and prev.state.get("ns_memory"):
                    initial_state["ns_memory"] = prev.state["ns_memory"]
            except Exception:
                pass  # memory inheritance is best-effort

        session = AssistantSession(
            user_id=user_id,
            title=title[:240] or "Untitled investigation",
            namespace_key=namespace_key,
            topic_keys=topic_keys,
            orientation=orientation,
            expertise_level=expertise_level,
            parent_session_id=parent_session_id,
            branch_from_message_id=branch_from_message_id,
            state=initial_state,
        )
        self._db.add(session)
        await self._db.flush()
        return session

    async def archive_session(self, user_id: UUID, session_id: UUID) -> bool:
        session = await self.get_session(user_id, session_id)
        if not session:
            return False
        session.status = AssistantSessionStatus.archived
        session.updated_at = datetime.now(timezone.utc)
        await self._db.flush()
        return True

    async def archive_all_sessions(self, user_id: UUID) -> int:
        """Archive every active session for a user. Returns count archived.

        Uses a single bulk UPDATE instead of loading every session row (with
        eager-loaded messages + tasks) into Python memory, which would OOM on
        users with large histories.
        """
        now = datetime.now(timezone.utc)
        result = await self._db.execute(
            update(AssistantSession)
            .where(
                AssistantSession.user_id == user_id,
                AssistantSession.status == AssistantSessionStatus.active,
            )
            .values(status=AssistantSessionStatus.archived, updated_at=now)
        )
        await self._db.flush()
        return result.rowcount

    async def patch_session_state(self, session_id: UUID | str, patch: dict) -> None:
        """Shallow-merge ``patch`` into ``AssistantSession.state`` (JSONB).

        Used by the orchestrator to persist rolling history summaries without
        touching any other state keys (memory, ns_memory, etc.).
        """
        from sqlalchemy.orm.attributes import flag_modified

        sid = UUID(str(session_id)) if not isinstance(session_id, UUID) else session_id
        result = await self._db.execute(select(AssistantSession).where(AssistantSession.id == sid))
        row = result.scalar_one_or_none()
        if row is None:
            return
        state = dict(row.state or {})
        state.update(patch)
        row.state = state
        flag_modified(row, "state")
        await self._db.flush()

    async def rename_session(self, user_id: UUID, session_id: UUID, title: str) -> bool:
        """Rename a session — used when the user wants to override the auto-derived title."""
        session = await self.get_session(user_id, session_id)
        if not session:
            return False
        clean = (title or "").strip()[:240]
        if not clean:
            return False
        session.title = clean
        session.updated_at = datetime.now(timezone.utc)
        # Mark the title as user-edited so the auto-refresh job stops touching
        # it. The metadata refresh checks ``state.title_user_edited`` and
        # leaves the title alone when it is true.
        try:
            from sqlalchemy.orm.attributes import flag_modified
            state = dict(session.state or {})
            state["title_user_edited"] = True
            state.pop("auto_title", None)
            session.state = state
            flag_modified(session, "state")
        except Exception:
            pass
        await self._db.flush()
        return True

    async def add_message(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        role: AssistantMessageRole | str,
        content: str,
        message_type: str = "text",
        citations: list[str] | None = None,
        artifact_refs: list[dict] | None = None,
        payload: dict | None = None,
    ) -> AssistantMessage:
        msg = AssistantMessage(
            session_id=session_id,
            user_id=user_id,
            role=AssistantMessageRole(role),
            content=content,
            message_type=message_type,
            citations=citations or [],
            artifact_refs=artifact_refs or [],
            payload=payload or {},
        )
        self._db.add(msg)
        await self.touch_session(session_id)
        await self._db.flush()
        return msg

    async def update_message(
        self,
        message_id: UUID,
        *,
        content: str | None = None,
        citations: list[str] | None = None,
        artifact_refs: list[dict] | None = None,
        payload: dict | None = None,
        message_type: str | None = None,
    ) -> AssistantMessage | None:
        result = await self._db.execute(
            select(AssistantMessage).where(AssistantMessage.id == message_id)
        )
        msg = result.scalar_one_or_none()
        if not msg:
            return None
        if content is not None:
            msg.content = content
        if citations is not None:
            msg.citations = citations
        if artifact_refs is not None:
            msg.artifact_refs = artifact_refs
        if payload is not None:
            msg.payload = payload
        if message_type is not None:
            msg.message_type = message_type
        await self.touch_session(msg.session_id)
        await self._db.flush()
        return msg

    async def create_task(
        self,
        *,
        job_id: str,
        session_id: UUID,
        user_id: UUID,
        assistant_message_id: UUID,
        task_type: str,
        title: str,
        namespace_key: str,
        progress: dict | None = None,
    ) -> AssistantTask:
        task = AssistantTask(
            job_id=job_id,
            session_id=session_id,
            user_id=user_id,
            assistant_message_id=assistant_message_id,
            task_type=task_type,
            title=title[:240],
            namespace_key=namespace_key,
            status=AssistantTaskStatus.pending,
            progress=progress or {"stage": "queued", "percent": 0},
        )
        self._db.add(task)
        await self.touch_session(session_id)
        await self._db.flush()
        return task

    async def get_task_by_job_id(self, user_id: UUID, job_id: str) -> AssistantTask | None:
        result = await self._db.execute(
            select(AssistantTask).where(
                AssistantTask.user_id == user_id,
                AssistantTask.job_id == job_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_task(self, task_id: UUID) -> AssistantTask | None:
        result = await self._db.execute(select(AssistantTask).where(AssistantTask.id == task_id))
        return result.scalar_one_or_none()

    async def list_tasks_for_user(self, user_id: UUID, limit: int = 100) -> list[AssistantTask]:
        result = await self._db.execute(
            select(AssistantTask)
            .where(AssistantTask.user_id == user_id)
            .order_by(AssistantTask.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())

    async def update_task(
        self,
        job_id: str,
        *,
        status: AssistantTaskStatus | str | None = None,
        progress: dict | None = None,
        result: dict | None = None,
        error: str | None = None,
        started: bool = False,
        completed: bool = False,
        cancel_requested: bool = False,
    ) -> AssistantTask | None:
        result_obj = await self._db.execute(select(AssistantTask).where(AssistantTask.job_id == job_id))
        task = result_obj.scalar_one_or_none()
        if not task:
            return None

        now = datetime.now(timezone.utc)
        if status is not None:
            task.status = AssistantTaskStatus(status)
        if progress is not None:
            task.progress = progress
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if started and task.started_at is None:
            task.started_at = now
        if completed:
            task.completed_at = task.completed_at or now
        if cancel_requested:
            task.cancel_requested_at = task.cancel_requested_at or now
        await self.touch_session(task.session_id)
        await self._db.flush()
        return task

    async def touch_session(self, session_id: UUID) -> None:
        result = await self._db.execute(
            select(AssistantSession).where(AssistantSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            session.updated_at = datetime.now(timezone.utc)

    # ── Steps ─────────────────────────────────────────────────────────────

    async def create_step(
        self,
        *,
        session_id: UUID,
        parent_message_id: UUID,
        job_id: str,
        step_index: int,
        tool_name: str,
        title: str,
        input_params: dict | None = None,
        parent_step_id: UUID | None = None,
    ) -> AssistantStep:
        step = AssistantStep(
            session_id=session_id,
            parent_message_id=parent_message_id,
            parent_step_id=parent_step_id,
            job_id=job_id,
            step_index=step_index,
            tool_name=tool_name,
            title=title[:240],
            status=AssistantStepStatus.pending,
            input_params=input_params or {},
        )
        self._db.add(step)
        await self._db.flush()
        return step

    async def update_step(
        self,
        step_id: UUID,
        *,
        status: AssistantStepStatus | str | None = None,
        progress: dict | None = None,
        output: dict | None = None,
        cost: dict | None = None,
        error: str | None = None,
        started: bool = False,
        completed: bool = False,
    ) -> AssistantStep | None:
        result = await self._db.execute(select(AssistantStep).where(AssistantStep.id == step_id))
        step = result.scalar_one_or_none()
        if not step:
            return None
        now = datetime.now(timezone.utc)
        if status is not None:
            step.status = AssistantStepStatus(status)
        if progress is not None:
            step.progress = progress
        if output is not None:
            step.output = output
        if cost is not None:
            step.cost = cost
        if error is not None:
            step.error = error
        if started and step.started_at is None:
            step.started_at = now
        if completed:
            step.completed_at = step.completed_at or now
        await self._db.flush()
        return step

    async def list_steps_for_message(self, message_id: UUID) -> list[AssistantStep]:
        result = await self._db.execute(
            select(AssistantStep)
            .where(AssistantStep.parent_message_id == message_id)
            .order_by(AssistantStep.step_index)
        )
        return list(result.scalars())

    async def list_steps_for_job(self, job_id: str) -> list[AssistantStep]:
        result = await self._db.execute(
            select(AssistantStep)
            .where(AssistantStep.job_id == job_id)
            .order_by(AssistantStep.step_index)
        )
        return list(result.scalars())

    async def list_steps_for_session(self, session_id: UUID, limit: int = 200) -> list[AssistantStep]:
        result = await self._db.execute(
            select(AssistantStep)
            .where(AssistantStep.session_id == session_id)
            .order_by(AssistantStep.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())

    # ── Artifacts ─────────────────────────────────────────────────────────

    async def create_artifact(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        kind: str,
        ref_id: str,
        title: str = "",
        href: str | None = None,
        preview: dict | None = None,
        producing_step_id: UUID | None = None,
        producing_message_id: UUID | None = None,
    ) -> AssistantArtifact:
        artifact = AssistantArtifact(
            session_id=session_id,
            user_id=user_id,
            kind=kind,
            ref_id=ref_id[:120],
            title=title[:240],
            href=href,
            preview=preview or {},
            producing_step_id=producing_step_id,
            producing_message_id=producing_message_id,
        )
        self._db.add(artifact)
        await self.touch_session(session_id)
        await self._db.flush()
        return artifact

    async def list_artifacts_for_session(
        self, session_id: UUID, limit: int = 100
    ) -> list[AssistantArtifact]:
        result = await self._db.execute(
            select(AssistantArtifact)
            .where(AssistantArtifact.session_id == session_id)
            .order_by(AssistantArtifact.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())

    # ── Attachments ───────────────────────────────────────────────────────

    async def create_attachment(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        kind: str,
        label: str = "",
        content: str | None = None,
        url: str | None = None,
        paper_id: UUID | None = None,
        message_id: UUID | None = None,
        metadata: dict | None = None,
    ) -> "AssistantAttachment":
        from app.models.assistant import AssistantAttachment

        att = AssistantAttachment(
            session_id=session_id,
            user_id=user_id,
            kind=kind[:40],
            label=label[:240],
            content=content,
            url=url[:2000] if url else None,
            paper_id=paper_id,
            message_id=message_id,
            metadata_=metadata or {},
        )
        self._db.add(att)
        await self.touch_session(session_id)
        await self._db.flush()
        return att

    async def list_attachments(self, session_id: UUID, limit: int = 100):
        from app.models.assistant import AssistantAttachment

        result = await self._db.execute(
            select(AssistantAttachment)
            .where(AssistantAttachment.session_id == session_id)
            .order_by(AssistantAttachment.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())

    async def delete_attachment(self, session_id: UUID, attachment_id: UUID) -> bool:
        from app.models.assistant import AssistantAttachment

        result = await self._db.execute(
            select(AssistantAttachment).where(
                AssistantAttachment.id == attachment_id,
                AssistantAttachment.session_id == session_id,
            )
        )
        att = result.scalar_one_or_none()
        if not att:
            return False
        await self._db.delete(att)
        await self.touch_session(session_id)
        await self._db.flush()
        return True
