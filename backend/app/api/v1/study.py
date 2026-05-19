"""Study router — SSE-streamed deep paper walkthrough + background job queue."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from jose import JWTError
from pydantic import BaseModel

from app.core.deps import CurrentUserID, DBSession, require_feature
from app.core.security import decode_access_token
from app.repositories.paper import PaperRepository
from app.workflows.study import get_user_jobs, queue_study, run_study, run_study_chat

router = APIRouter(
    prefix="/study",
    tags=["study"],
    dependencies=[Depends(require_feature("study_mode_enabled"))],
)


async def _user_id_from_query(token: str = Query(...)) -> UUID:
    """Auth dependency for SSE endpoints where EventSource can't set headers."""
    from app.core.tracking import current_user_id as _ctx
    try:
        payload = decode_access_token(token)
        user_id_str: str | None = payload.get("sub")
        if not user_id_str:
            raise ValueError("missing sub")
        uid = UUID(user_id_str)
        _ctx.set(uid)  # so token tracking attributes LLM calls during SSE
        return uid
    except (JWTError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


SSEUserID = Annotated[UUID, Depends(_user_id_from_query)]


@router.get("/jobs", response_model=list[dict])
async def list_jobs(user_id: CurrentUserID):
    """Return all study jobs for the current user."""
    return get_user_jobs(str(user_id))


class QueueRequest(BaseModel):
    """Request body for POST /study/{paper_id}/queue."""

    expertise_level: str = "practitioner"


@router.post("/{paper_id}/queue")
async def queue_study_job(
    paper_id: UUID,
    body: QueueRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Queue a Study Mode job for the given paper and return its job ID.

    Args:
        paper_id: UUID of the paper to study.
        body: Queue request containing the desired ``expertise_level``.
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        A dict with ``job_id`` and ``status`` set to ``"pending"``.

    Raises:
        HTTPException: 404 if the paper is not found.
    """
    paper_repo = PaperRepository(db)
    paper = await paper_repo.get_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    job_id = queue_study(paper_id, body.expertise_level, user_id, paper.title)
    return {"job_id": job_id, "status": "pending"}


class ChatRequest(BaseModel):
    """Request body for POST /study/{paper_id}/chat."""

    message: str
    expertise_level: str = "practitioner"
    history: list[dict] = []


@router.post("/{paper_id}/chat", response_class=StreamingResponse)
async def chat_study(
    paper_id: UUID,
    body: ChatRequest,
    user_id: CurrentUserID,
):
    """Stream a Q&A response grounded in the paper's cached study content."""
    async def event_generator():
        """Yield SSE chunks from the study chat workflow."""
        async for chunk in run_study_chat(
            paper_id, body.expertise_level, body.message, body.history
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{paper_id}", response_class=StreamingResponse)
async def stream_study(
    paper_id: UUID,
    user_id: SSEUserID,
    expertise_level: str = "practitioner",
):
    """Start Study Mode — returns an SSE stream of sections."""
    async def event_generator():
        """Yield SSE chunks from the full study workflow."""
        async for chunk in run_study(paper_id, expertise_level, user_id):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
