"""RAG Chat router — grounded, cited, scoped knowledge chat (SSE streaming)."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.deps import CurrentUserID
from app.schemas import ChatRequest
from app.workflows.rag import run_rag_stream

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_class=StreamingResponse)
async def chat(body: ChatRequest, user_id: CurrentUserID):
    """Stream a RAG-grounded chat response as server-sent events.

    Delegates to ``run_rag_stream`` which retrieves relevant chunks from the
    knowledge graph and bookmarks, then streams the LLM response token by token.

    Args:
        body: Chat request containing the user query and namespace scope.
        user_id: UUID of the authenticated user.

    Returns:
        A ``StreamingResponse`` with ``text/event-stream`` content type.
    """
    async def event_gen():
        """Yield SSE chunks from the RAG streaming workflow."""
        async for chunk in run_rag_stream(user_id, body.namespace_key, body.query):
            yield chunk

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
