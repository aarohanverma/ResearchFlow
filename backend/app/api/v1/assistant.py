"""Research Assistant API — persistent AI-native research workspace."""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from app.assistant.events import get_event_bus
from app.assistant.tools import describe_for_planner
from app.core.deps import CurrentUserID, DBSession
from app.repositories.assistant import AssistantRepository
from app.schemas import (
    ArxivImportRequest,
    ArxivSearchRequest,
    AssistantArtifactResponse,
    AssistantAttachmentCreateRequest,
    AssistantAttachmentResponse,
    AssistantBranchRequest,
    AssistantMessageRequest,
    AssistantSessionCreateRequest,
    AssistantSessionRenameRequest,
    AssistantSessionResponse,
    AssistantStepResponse,
    AssistantSubmitResponse,
    AssistantTaskResponse,
    AssistantToolDescriptor,
)
from app.services import research_assistant as assistant_service
from app.services.arxiv_import import ArxivImportService

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.get("/sessions", response_model=list[AssistantSessionResponse])
async def list_sessions(
    user_id: CurrentUserID,
    db: DBSession,
    limit: int = Query(default=50, le=100),
    namespace_key: str | None = Query(default=None),
):
    """List persistent assistant sessions for the current user, optionally filtered by namespace."""
    repo = AssistantRepository(db)
    sessions = await repo.list_sessions(user_id, limit=limit, namespace_key=namespace_key)
    return [AssistantSessionResponse.model_validate(s) for s in sessions]


@router.post("/sessions", response_model=AssistantSessionResponse, status_code=201)
async def create_session(body: AssistantSessionCreateRequest, user_id: CurrentUserID, db: DBSession):
    """Create a new persistent research workspace session."""
    sid = await assistant_service.create_session(
        user_id=user_id,
        namespace_key=body.namespace_key,
        topic_keys=body.topic_keys,
        title=body.title,
    )
    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, sid)
    if not session:
        raise HTTPException(status_code=500, detail="Session creation failed")
    return AssistantSessionResponse.model_validate(session)


@router.get("/sessions/{session_id}", response_model=AssistantSessionResponse)
async def get_session(session_id: UUID, user_id: CurrentUserID, db: DBSession):
    """Return a session with messages and tasks."""
    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return AssistantSessionResponse.model_validate(session)


@router.get("/sessions/{session_id}/export")
async def export_session(
    session_id: UUID,
    user_id: CurrentUserID,
    db: DBSession,
    format: str = Query(default="markdown", pattern="^(markdown|json)$"),
):
    """Export a session as Markdown or JSON for sharing."""
    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    title = (session.title or "Research Session").strip()
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)[:50].strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if format == "json":
        data = AssistantSessionResponse.model_validate(session).model_dump(mode="json")
        import json as _json
        return Response(
            content=_json.dumps(data, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{safe_title or session_id}.json"'},
        )

    # Markdown export
    lines: list[str] = [f"# {title}\n"]
    lines.append(f"**Namespace:** {session.namespace_key}  ")
    if session.topic_keys:
        lines.append(f"**Topics:** {', '.join(session.topic_keys)}  ")
    lines.append(f"**Exported:** {now}\n")
    lines.append("---\n")

    sorted_msgs = sorted(session.messages, key=lambda m: m.created_at)
    for msg in sorted_msgs:
        if msg.role.value == "system":  # type: ignore[union-attr]
            continue
        role_label = "You" if msg.role.value == "user" else "Research Assistant"  # type: ignore[union-attr]
        lines.append(f"### {role_label}\n")
        content = (msg.content or "").strip()
        if not content:
            blocks = (msg.payload or {}).get("blocks", [])
            for b in blocks:
                if isinstance(b, dict) and b.get("kind") == "text":
                    content += b.get("content", "")
        if content:
            lines.append(content)
        lines.append("\n---\n")

    markdown = "\n".join(lines)
    return PlainTextResponse(
        content=markdown,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_title or session_id}.md"',
            "Content-Type": "text/markdown; charset=utf-8",
        },
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def archive_session(session_id: UUID, user_id: CurrentUserID, db: DBSession):
    """Archive a session without deleting its persisted data."""
    repo = AssistantRepository(db)
    ok = await repo.archive_session(user_id, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.commit()


@router.patch("/sessions/{session_id}/title", response_model=AssistantSessionResponse)
async def rename_session(
    session_id: UUID,
    body: AssistantSessionRenameRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Rename a session — overrides the auto-derived title."""
    repo = AssistantRepository(db)
    ok = await repo.rename_session(user_id, session_id, body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.commit()
    session = await repo.get_session(user_id, session_id)
    return AssistantSessionResponse.model_validate(session)


@router.post("/sessions/clear", status_code=200)
async def clear_all_sessions(user_id: CurrentUserID, db: DBSession):
    """Bulk-archive every active session for the user.

    Soft delete — the rows remain queryable for audit / "show archived" UIs;
    only the active list is cleared. Returns the count for UI feedback.
    """
    repo = AssistantRepository(db)
    archived = await repo.archive_all_sessions(user_id)
    await db.commit()
    return {"archived": archived}


@router.post("/sessions/{session_id}/branch", response_model=AssistantSessionResponse, status_code=201)
async def branch_session(
    session_id: UUID,
    body: AssistantBranchRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Branch a research investigation from an existing session/message."""
    source_repo = AssistantRepository(db)
    source_session = await source_repo.get_session(user_id, session_id)
    if not source_session:
        raise HTTPException(status_code=404, detail="Source session not found")
    child_id = await assistant_service.branch_session(
        user_id=user_id,
        source_session_id=session_id,
        from_message_id=body.from_message_id,
        title=body.title,
    )
    if not child_id:
        raise HTTPException(
            status_code=422,
            detail="Cannot branch: maximum nesting depth (3 levels) reached.",
        )
    repo = AssistantRepository(db)
    child = await repo.get_session(user_id, child_id)
    if not child:
        raise HTTPException(status_code=500, detail="Branch creation failed")
    return AssistantSessionResponse.model_validate(child)


@router.post("/sessions/{session_id}/messages", response_model=AssistantSubmitResponse, status_code=202)
async def submit_message(
    session_id: UUID,
    body: AssistantMessageRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Queue an assistant orchestration turn and return immediately."""
    try:
        user_msg_id, assistant_msg_id, job_id = await assistant_service.submit_turn(
            user_id=user_id,
            session_id=session_id,
            content=body.content,
            namespace_key=body.namespace_key,
            topic_keys=body.topic_keys,
            attachments=body.attachments,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    user_msg = next(m for m in session.messages if m.id == user_msg_id)
    assistant_msg = next(m for m in session.messages if m.id == assistant_msg_id)
    task = next(t for t in session.tasks if t.job_id == job_id)
    return AssistantSubmitResponse(
        session=AssistantSessionResponse.model_validate(session),
        user_message=user_msg,
        assistant_message=assistant_msg,
        task=task,
    )


@router.post("/tasks/{job_id}/cancel", status_code=200)
async def cancel_task(job_id: str, user_id: CurrentUserID):
    """Cancel a running assistant task."""
    ok = await assistant_service.cancel_task(user_id, job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "cancelled", "job_id": job_id}


@router.get("/tasks", response_model=list[AssistantTaskResponse])
async def list_tasks(user_id: CurrentUserID, db: DBSession):
    """List assistant tasks for the user."""
    repo = AssistantRepository(db)
    return [AssistantTaskResponse.model_validate(t) for t in await repo.list_tasks_for_user(user_id)]


@router.get("/jobs")
async def assistant_jobs(user_id: CurrentUserID, db: DBSession):
    """Notification-panel view of assistant jobs."""
    repo = AssistantRepository(db)
    tasks = await repo.list_tasks_for_user(user_id)
    return {
        "jobs": [
            {
                "kind": "assistant",
                "job_id": t.job_id,
                "task_id": str(t.id),
                "session_id": str(t.session_id),
                "assistant_message_id": str(t.assistant_message_id) if t.assistant_message_id else None,
                "title": t.title,
                "status": t.status.value if hasattr(t.status, "value") else str(t.status),
                "namespace_key": t.namespace_key,
                "summary": (t.progress or {}).get("summary") or t.error,
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "href": f"/assistant?session={t.session_id}",
            }
            for t in tasks[:100]
        ],
        "total": len(tasks),
    }


@router.get("/sessions/{session_id}/steps", response_model=list[AssistantStepResponse])
async def list_session_steps(
    session_id: UUID,
    user_id: CurrentUserID,
    db: DBSession,
    limit: int = Query(default=200, le=500),
):
    """Reasoning-tree view: every step executed inside a session, newest first."""
    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    steps = await repo.list_steps_for_session(session_id, limit=limit)
    return [AssistantStepResponse.model_validate(s) for s in steps]


@router.get("/messages/{message_id}/steps", response_model=list[AssistantStepResponse])
async def list_message_steps(message_id: UUID, user_id: CurrentUserID, db: DBSession):
    """Steps that produced a single assistant message — used for inline reasoning rendering."""
    repo = AssistantRepository(db)
    steps = await repo.list_steps_for_message(message_id)
    if steps:
        # Authorization: confirm the parent session belongs to the user.
        session = await repo.get_session(user_id, steps[0].session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Message not found")
    return [AssistantStepResponse.model_validate(s) for s in steps]


@router.get("/sessions/{session_id}/artifacts", response_model=list[AssistantArtifactResponse])
async def list_session_artifacts(
    session_id: UUID,
    user_id: CurrentUserID,
    db: DBSession,
    limit: int = Query(default=100, le=500),
):
    """Artifacts produced inside a session (study summaries, ideas, podcasts, …)."""
    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    artifacts = await repo.list_artifacts_for_session(session_id, limit=limit)
    return [AssistantArtifactResponse.model_validate(a) for a in artifacts]


@router.get("/tools", response_model=list[AssistantToolDescriptor])
async def list_tools(user_id: CurrentUserID):  # noqa: ARG001 — auth-gated catalogue
    """Schema-only view of the registered assistant tool surface."""
    return [AssistantToolDescriptor(**t) for t in describe_for_planner()]


# ── Attachments ───────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/attachments", response_model=list[AssistantAttachmentResponse])
async def list_attachments(
    session_id: UUID,
    user_id: CurrentUserID,
    db: DBSession,
    limit: int = Query(default=100, le=500),
):
    """List session-scoped attachments (notes/URLs/papers/PDFs/images)."""
    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    rows = await repo.list_attachments(session_id, limit=limit)
    return [AssistantAttachmentResponse.model_validate(r) for r in rows]


@router.post(
    "/sessions/{session_id}/attachments",
    response_model=AssistantAttachmentResponse,
    status_code=201,
)
async def create_attachment(
    session_id: UUID,
    body: AssistantAttachmentCreateRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Attach a session-scoped note / URL / paper-ref / PDF / image.

    Validates ``kind`` against a small allowlist and refuses payloads that
    don't carry the field expected for that kind (a URL kind needs ``url``,
    a note needs ``content``, a paper_ref needs ``paper_id``).
    """
    allowed = {"note", "url", "paper_ref", "pdf", "image"}
    if body.kind not in allowed:
        raise HTTPException(status_code=400, detail=f"unsupported attachment kind: {body.kind}")
    if body.kind == "url" and not body.url:
        raise HTTPException(status_code=400, detail="url attachment requires a `url` field")
    if body.kind == "note" and not body.content:
        raise HTTPException(status_code=400, detail="note attachment requires `content`")
    if body.kind == "paper_ref" and not body.paper_id:
        raise HTTPException(status_code=400, detail="paper_ref attachment requires `paper_id`")

    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    att = await repo.create_attachment(
        session_id=session_id,
        user_id=user_id,
        kind=body.kind,
        label=body.label,
        content=body.content,
        url=body.url,
        paper_id=body.paper_id,
        metadata=body.metadata,
    )
    await db.commit()
    return AssistantAttachmentResponse.model_validate(att)


@router.delete("/sessions/{session_id}/attachments/{attachment_id}", status_code=204)
async def delete_attachment(
    session_id: UUID,
    attachment_id: UUID,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Remove a session-scoped attachment."""
    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    ok = await repo.delete_attachment(session_id, attachment_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Attachment not found")
    await db.commit()


# ── Multimodal uploads (PDF / image) ──────────────────────────────────────────

# Conservative size cap — stops a memory-bloating upload from taking down the
# parser. ResearchFlow's PDF chain spikes RAM during Marker/Docling parses.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
_PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}
_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".html", ".htm", ".xml", ".svg",
    ".tex", ".bib", ".log",
}


@router.post(
    "/sessions/{session_id}/attachments/upload",
    response_model=AssistantAttachmentResponse,
    status_code=201,
)
async def upload_attachment(
    session_id: UUID,
    user_id: CurrentUserID,
    db: DBSession,
    file: UploadFile = File(...),
):
    """Upload any file and persist it as a session attachment.

    PDFs flow through the platform's parser chain (Marker → Docling → Gemini
    Vision) so extracted text becomes RAG context. Images flow through Gemini
    Vision for caption + OCR. Text-based files (code, markdown, CSV, JSON,
    YAML, etc.) are decoded as UTF-8 directly. Binary files that don't match
    any known type are stored as-is with a parse_error hint.

    Failures during parsing never fail the upload — the row is always
    persisted; ``parse_error`` in metadata surfaces a hint to the frontend.
    """
    repo = AssistantRepository(db)
    session = await repo.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)} bytes > {_MAX_UPLOAD_BYTES} cap)",
        )

    content_type = (file.content_type or "").lower()
    filename = file.filename or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    is_pdf = content_type in _PDF_MIME_TYPES or ext == ".pdf"
    is_image = content_type in _IMAGE_MIME_TYPES or ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    is_text = ext in _TEXT_EXTENSIONS or content_type.startswith("text/")
    is_docx = ext == ".docx" or content_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }

    extracted_text: str | None = None
    metadata: dict = {
        "filename": filename,
        "content_type": content_type,
        "byte_size": len(raw),
    }
    attachment_kind = "file"

    if is_pdf:
        attachment_kind = "pdf"
        try:
            from app.adapters.pdf import parse_with_fallback

            parsed = await parse_with_fallback(raw)
            metadata.update({
                "parser": parsed.parser_name,
                "parser_fallback_used": parsed.fallback_used,
                "parser_confidence": parsed.parser_confidence,
                "parse_duration_ms": parsed.parse_duration_ms,
                "section_count": len(parsed.sections),
                "title_extracted": parsed.title or None,
            })
            chunks: list[str] = []
            if parsed.title:
                chunks.append(f"# {parsed.title}")
            if parsed.abstract:
                chunks.append(f"## Abstract\n{parsed.abstract}")
            for section in parsed.sections:
                heading = getattr(section, "heading", None) or getattr(section, "title", None)
                content = getattr(section, "content", None) or getattr(section, "text", None) or ""
                if heading:
                    chunks.append(f"## {heading}\n{content}")
                elif content:
                    chunks.append(content)
            extracted_text = "\n\n".join(c for c in chunks if c).strip() or None
        except Exception as exc:
            metadata["parse_error"] = str(exc)[:240]

    elif is_image:
        attachment_kind = "image"
        try:
            extracted_text, vision_meta = await _extract_image_text(raw, content_type or "image/png")
            metadata.update(vision_meta)
        except Exception as exc:
            metadata["parse_error"] = str(exc)[:240]

    elif is_docx:
        try:
            extracted_text = _extract_docx_text(raw)
            metadata["parser"] = "python-docx"
        except Exception as exc:
            metadata["parse_error"] = str(exc)[:240]

    elif is_text:
        try:
            extracted_text = raw.decode("utf-8", errors="replace")
            metadata["parser"] = "utf8-decode"
        except Exception as exc:
            metadata["parse_error"] = str(exc)[:240]

    else:
        # Unknown binary — try UTF-8 decode as last resort
        try:
            decoded = raw.decode("utf-8", errors="strict")
            extracted_text = decoded
            metadata["parser"] = "utf8-fallback"
        except UnicodeDecodeError:
            metadata["parse_error"] = f"Binary file type ({ext or content_type}) — text extraction not supported"

    att = await repo.create_attachment(
        session_id=session_id,
        user_id=user_id,
        kind=attachment_kind,
        label=metadata.get("title_extracted") or filename,
        content=extracted_text,
        url=None,
        metadata=metadata,
    )
    await db.commit()
    return AssistantAttachmentResponse.model_validate(att)


def _extract_docx_text(data: bytes) -> str:
    """Extract plain text from a .docx file using python-docx."""
    import io
    try:
        import docx  # python-docx
    except ImportError:
        # Graceful: extract XML text manually without the library
        import zipfile, re
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
        return re.sub(r"<[^>]+>", " ", xml).strip()
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


async def _extract_image_text(image_bytes: bytes, content_type: str) -> tuple[str | None, dict]:
    """Run an image through the configured vision LLM for caption + OCR.

    Best-effort: returns (None, {}) when no vision-capable provider is
    configured, instead of raising — the upload still succeeds and the
    user gets the file as an attachment, just without extracted text.
    """
    try:
        # Gemini is the project's vision-capable adapter; reusing the same
        # path the PDF Gemini-Vision fallback uses keeps dependency surface
        # minimal and ensures consistent behavior across the app.
        from app.adapters.llm import get_llm_adapter
        import base64

        llm = get_llm_adapter()
        if not hasattr(llm, "complete"):
            return None, {}
        b64 = base64.b64encode(image_bytes).decode("ascii")
        # Generic multimodal message — providers without vision support will
        # raise here; we catch and degrade rather than failing the upload.
        prompt = (
            "Describe this image in 2-3 sentences for a research workspace. "
            "Then transcribe any visible text verbatim and list any visible "
            "concepts/terms that could anchor a literature search."
        )
        try:
            res = await llm.complete(
                [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{b64}"}},
                    ],
                }],
                llm.quality_model,
                max_tokens=600,
            )
            text = (res.text or "").strip()
            return (text or None), {"vision_provider": llm.provider_id if hasattr(llm, "provider_id") else "unknown"}
        except Exception:
            # Fallback: just pass the prompt as plain text — many providers
            # silently ignore the image. Returning None tells the frontend
            # the image was stored but not analyzed.
            return None, {"vision_provider": "unavailable"}
    except Exception:
        return None, {}


@router.get("/seeds")
async def get_seed_questions(
    namespace_key: str = Query(default=""),
    user_id: CurrentUserID = None,
):
    """Return 4 namespace-specific seed questions for the assistant empty state.

    Checks a static map first; generates via LLM for unknown namespaces with
    a 2-second timeout and graceful fallback.
    """
    questions = _seeds_for_namespace(namespace_key)
    return {"questions": questions, "namespace_key": namespace_key}


_STATIC_SEEDS: dict[str, list[str]] = {
    "cs.AI":  ["What are the frontier directions in mechanistic interpretability?", "Compare RAG vs long-context LLMs for knowledge-intensive tasks.", "Help me understand agentic AI — where do I begin?", "What are the open problems in AI alignment research?"],
    "cs.LG":  ["What are leading approaches for sample-efficient RL?", "Compare transformers vs state-space models for sequences.", "What is the current state of neural scaling laws?", "Help me find papers on self-supervised learning for tabular data."],
    "cs.CV":  ["What are frontier directions in 3D scene understanding?", "Compare CLIP, ALIGN, and SigLIP vision-language models.", "Help me explore diffusion models for controllable image generation.", "What recent work addresses long-tailed recognition?"],
    "cs.CL":  ["What is the state of low-resource machine translation?", "Compare instruction tuning vs RLHF vs DPO for alignment.", "Help me find papers on structured prediction and NLP.", "What are the open problems in multilingual NLP?"],
    "cs.RO":  ["What are leading methods for robot learning from demonstration?", "How are foundation models applied to embodied AI?", "Compare sim-to-real transfer approaches.", "What is the state of autonomous vehicle perception?"],
    "cs.CR":  ["What are frontier directions in post-quantum cryptography?", "Help me understand differential privacy in machine learning.", "What are the state-of-the-art adversarial attack defenses?", "Compare federated learning approaches for privacy-preserving ML."],
    "cs.IR":  ["What are the latest advances in dense retrieval?", "Compare BM25 vs neural retrieval for question answering.", "Help me understand multi-modal information retrieval.", "What are open problems in personalized recommendation systems?"],
    "cs.SE":  ["What are frontier directions in automated program repair?", "How is LLM applied to code generation and synthesis?", "Compare static vs dynamic analysis approaches for bug finding.", "What is the state of formal verification for real-world software?"],
    "cs.DB":  ["What are the frontier directions in learned database systems?", "Compare NewSQL vs NoSQL for OLTP workloads.", "Help me understand cardinality estimation with ML.", "What are recent advances in query optimization?"],
    "cs.DC":  ["What are the frontier approaches in Byzantine fault-tolerant consensus?", "Compare Paxos vs Raft vs PBFT for distributed systems.", "Help me understand RDMA-based distributed memory systems.", "What are recent advances in geo-distributed computing?"],
    "quant-ph": ["What are the most promising approaches to quantum error correction?", "Help me understand the quantum advantage for optimization.", "Compare superconducting vs photonic vs trapped-ion qubits.", "What is the state of quantum machine learning research?"],
    "q-bio":  ["What are state-of-the-art methods for protein structure prediction beyond AlphaFold?", "How is deep learning used for drug discovery?", "Help me understand graph neural networks for biological networks.", "What are open problems in computational genomics?"],
    "math":   ["What are recent breakthroughs in extremal combinatorics?", "Help me understand connections between category theory and type theory.", "Compare approaches to automated theorem proving.", "What is the frontier of Langlands program research?"],
    "stat":   ["What are frontier directions in Bayesian deep learning?", "Help me understand modern causal discovery approaches.", "Compare frequentist vs Bayesian high-dimensional inference.", "What is the state of conformal prediction research?"],
    "econ":   ["What are recent developments in algorithmic mechanism design?", "Help me understand causal inference in observational studies.", "Compare structural vs reduced-form approaches in empirical economics.", "What are frontier topics in market microstructure?"],
    "physics": ["What are promising approaches to room-temperature superconductivity?", "Help me understand recent quantum computing hardware experiments.", "What is the state of dark matter detection?", "How is ML applied to high-energy physics data?"],
    "astro-ph": ["What are the latest results in multi-messenger astronomy?", "Help me understand the tension in the Hubble constant measurements.", "Compare approaches to gravitational wave detection.", "What are frontier questions in exoplanet atmosphere characterization?"],
    "hep-th": ["What is the current state of the swampland program?", "Help me understand recent advances in string theory and holography.", "What are the open problems in quantum gravity?", "Compare approaches to black hole information paradox."],
    "cond-mat": ["What are the frontier topics in topological phases of matter?", "Help me understand recent progress in high-temperature superconductivity.", "Compare different approaches to quantum simulation.", "What is the state of quantum spin liquid research?"],
}


async def _seeds_for_namespace(namespace_key: str) -> list[str]:
    """Return 4 seed questions for the given namespace, generating via LLM if needed."""
    # Direct match
    if namespace_key in _STATIC_SEEDS:
        return _STATIC_SEEDS[namespace_key]
    # Prefix match (e.g. "cond-mat.mtrl-sci" → "cond-mat")
    prefix = namespace_key.split(".")[0]
    for k, v in _STATIC_SEEDS.items():
        if k.split(".")[0] == prefix:
            return v
    # LLM generation for unknown namespaces with tight timeout
    try:
        import asyncio
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()
        prompt = (
            f"Generate exactly 4 short, specific, research-level starter questions for the arXiv namespace '{namespace_key}'. "
            "Each question should invite deep exploration of a frontier topic. "
            "Return ONLY a JSON array of 4 strings, no explanation."
        )
        res = await asyncio.wait_for(
            llm.complete([{"role": "user", "content": prompt}], llm.cheap_model, max_tokens=300),
            timeout=3.0,
        )
        import json as _json
        text = (res.text or "").strip()
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            parsed = _json.loads(text[start:end+1])
            if isinstance(parsed, list) and len(parsed) >= 2:
                return [str(q) for q in parsed[:4]]
    except Exception:
        pass
    # Generic fallback
    return [
        "What are the most important recent papers in this research area?",
        "Help me understand the key open problems and frontier directions here.",
        "Compare the leading methodologies and their trade-offs in this field.",
        "I want to start a research project — help me map the landscape.",
    ]


@router.get("/tasks/{job_id}/stream")
async def stream_task_events(
    job_id: str,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Stream typed AssistantEvent objects for a running turn over SSE.

    Subscribers receive the buffered event history for the job (plan, started
    steps, progress updates) plus all subsequent events until the bus closes.
    Heartbeats every 15 s keep proxies / load balancers from severing the
    connection. Authorized via standard JWT — the caller must own the task.
    """
    repo = AssistantRepository(db)
    task = await repo.get_task_by_job_id(user_id, job_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    bus = get_event_bus()
    queue = bus.subscribe(job_id)

    async def event_gen():
        try:
            yield "event: hello\ndata: {}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Heartbeat keeps proxies (nginx / cloud LBs) from idling out.
                    yield ": heartbeat\n\n"
                    continue
                payload = json.dumps(event.to_json(), default=str)
                yield f"event: {event.kind}\ndata: {payload}\n\n"
                if event.kind in {"task_completed", "task_failed", "task_cancelled"}:
                    break
        finally:
            bus.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/arxiv/search")
async def arxiv_search(body: ArxivSearchRequest, user_id: CurrentUserID, db: DBSession):
    """Search arXiv through MCP with namespace-aware fallback."""
    svc = ArxivImportService(db)
    results = await svc.search(
        body.query,
        namespace_keys=body.namespace_keys or None,
        max_results=body.max_results,
    )
    return {"results": results, "total": len(results)}


@router.post("/arxiv/import")
async def arxiv_import(body: ArxivImportRequest, user_id: CurrentUserID, db: DBSession):
    """Import selected arXiv papers into the active feed namespace."""
    svc = ArxivImportService(db)
    new_papers, skipped = await svc.import_raw_papers(body.papers, namespace_key=body.namespace_key)
    return {
        "imported": len(new_papers),
        "skipped": skipped,
        "paper_ids": [str(p.id) for p in new_papers],
        "namespace_key": body.namespace_key,
    }
