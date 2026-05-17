"""Pydantic v2 schemas — request/response contracts for all API routes."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """Request body for POST /auth/register."""

    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1, max_length=100)


class LoginRequest(BaseModel):
    """Request body for POST /auth/login."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Response body containing a JWT bearer token."""

    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Public user profile returned by GET /auth/me."""

    id: UUID
    email: str
    display_name: str
    expertise_level: str
    orientation: str
    onboarding_complete: bool

    model_config = {"from_attributes": True}


# ── Onboarding ────────────────────────────────────────────────────────────────

class OnboardingRequest(BaseModel):
    """Request body for POST /settings/onboarding."""

    subjects: list[str]
    topics: list[str]          # list of "Subject:Topic" strings
    expertise_level: str
    orientation: str
    notify_potd: bool = True
    notify_digest: bool = True
    notify_breakthrough: bool = True


# ── Papers ────────────────────────────────────────────────────────────────────

class PaperResponse(BaseModel):
    """Full paper detail returned by paper-fetching endpoints."""

    id: UUID
    external_id: str
    namespace_key: str
    # All topic memberships visible to the user, populated by dedup passes.
    # When unset, the client treats the row's primary namespace_key as the
    # sole membership.
    namespace_keys: list[str] | None = None
    title: str
    authors: list[str]
    abstract: str
    source_url: str
    pdf_url: str | None
    published_at: datetime | None
    key_concepts: list[str]
    methods_used: list[str]
    implications: str | None
    novelty_score: float
    relevance_score: float
    is_breakthrough: bool
    tldr: str | None = None
    ingested_at: datetime
    is_manually_imported: bool = False

    model_config = {"from_attributes": True}


class FeedPaperResponse(BaseModel):
    """A single scored paper entry in the personalised feed."""

    paper: PaperResponse
    score: float
    why_tag: str


class FeedResponse(BaseModel):
    """Paginated feed response for GET /feed."""

    papers: list[FeedPaperResponse]
    total: int
    namespace_key: str


class FeedbackRequest(BaseModel):
    """Request body for POST /feed/feedback."""

    paper_id: UUID
    signal: str = Field(pattern="^(like|dismiss|more_like_this)$")


# ── Bookmark Folders ──────────────────────────────────────────────────────────

class FolderCreateRequest(BaseModel):
    """Request body for creating or renaming a bookmark folder."""

    name: str = Field(min_length=1, max_length=200)
    color: str | None = None


class FolderResponse(BaseModel):
    """Bookmark folder detail including the number of bookmarks it contains."""

    id: UUID
    name: str
    color: str | None
    created_at: datetime
    bookmark_count: int = 0

    model_config = {"from_attributes": True}


# ── Bookmarks ─────────────────────────────────────────────────────────────────

class BookmarkRequest(BaseModel):
    """Request body for POST /bookmarks."""

    paper_id: UUID
    note: str | None = None
    folder_ids: list[UUID] = []


class BookmarkResponse(BaseModel):
    """Bookmark detail including optional folder memberships and paper data."""

    id: UUID
    paper_id: UUID
    folder_ids: list[UUID] = []
    note: str | None
    created_at: datetime
    paper: PaperResponse | None = None

    model_config = {"from_attributes": True}


# ── Study ─────────────────────────────────────────────────────────────────────

class StudyRequest(BaseModel):
    """Request body for queueing a Study Mode job."""

    paper_id: UUID
    expertise_level: str = Field(pattern="^(newcomer|practitioner|expert)$")


# ── RAG Chat ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    query: str = Field(min_length=3, max_length=2000)
    namespace_key: str


class ChatResponse(BaseModel):
    """Non-streaming chat response with grounding citations."""

    answer: str
    citation_paper_ids: list[str]
    highlight_node_ids: list[str]
    scope_level: str


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class GraphResponse(BaseModel):
    """Subgraph payload returned by GET /graph."""

    nodes: list[dict]
    edges: list[dict]


# ── Genie ─────────────────────────────────────────────────────────────────────

class GenieRequest(BaseModel):
    """Request body for POST /genie/synthesize.

    Manual mode allows up to 10 seed elements so users can provide richer
    context.  Auto-batch groups are capped at 5 papers by the pairing logic.
    """

    seed_element_ids: list[str] = Field(min_length=2, max_length=10)
    namespace_key: str | None = None
    sem_threshold: float = Field(default=0.25, ge=0.05, le=0.95)


class SourcePaperInfo(BaseModel):
    """Minimal paper metadata embedded in ``IdeaCapsuleResponse.source_papers``."""

    id: str
    title: str
    authors: list[str]
    year: int | None
    url: str


class IdeaCapsuleResponse(BaseModel):
    """Full idea capsule detail returned by Genie capsule endpoints."""

    id: UUID
    title: str
    hypothesis: str
    rationale: str
    mechanism: str | None
    predicted_outcome: str | None
    experimental_design: str | None
    anti_finding: str | None
    risks_and_limitations: str | None
    open_questions: str | None
    novelty_score: float
    feasibility_score: float
    impact_score: float
    diagrams: list[dict]
    poc_code: str | None
    seed_element_ids: list[str]
    status: str
    is_scout_generated: bool = False
    source_mode: str = "manual"   # "manual" | "auto" | "query"
    source_query: str | None = None
    deep_dive_content: str | None = None
    deep_dive_status: str = "none"
    created_at: datetime
    source_papers: list[SourcePaperInfo] = []

    model_config = {"from_attributes": True}


class IdeaCapsuleListItem(BaseModel):
    """Slim capsule summary for list views.

    Drops the heavy long-form fields (mechanism, predicted_outcome,
    experimental_design, anti_finding, risks_and_limitations, diagrams,
    poc_code, deep_dive_content) which are only needed on the detail page.
    Removing them slashes the /capsules payload size by 10-50x for users
    with many ideas — the Genie Ideas list goes from "skeleton shimmers
    for two seconds" to "instant".
    """

    id: UUID
    title: str
    hypothesis: str
    open_questions: str | None = None
    novelty_score: float
    feasibility_score: float
    impact_score: float
    status: str
    is_scout_generated: bool = False
    source_mode: str = "manual"
    source_query: str | None = None
    deep_dive_status: str = "none"
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Settings ──────────────────────────────────────────────────────────────────

class ProviderSettingsRequest(BaseModel):
    """Request body for PATCH /settings/provider."""

    llm_provider: str | None = None
    cheap_model: str | None = None
    quality_model: str | None = None
    reasoning_model: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None


class NotificationSettingsRequest(BaseModel):
    """Request body for PATCH /settings/notifications."""

    notify_potd: bool | None = None
    notify_digest: bool | None = None
    notify_breakthrough: bool | None = None


# ── Search ────────────────────────────────────────────────────────────────────

class SearchResultItem(BaseModel):
    """A single result item in a hybrid search response."""

    paper_id: UUID
    external_id: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str]
    namespace_key: str
    # All topic memberships matched by the user's current scope. Populated by
    # the dedup pass in the search endpoint so the UI can render all relevant
    # topic tags on a single card.
    namespace_keys: list[str] | None = None
    source_url: str
    pdf_url: str | None
    novelty_score: float
    relevance_score: float
    is_breakthrough: bool
    is_manually_imported: bool = False
    key_concepts: list[str] | None = None
    methods_used: list[str] | None = None
    implications: str | None = None
    published_at: datetime | None = None
    ingested_at: datetime | None = None
    tldr: str | None = None
    search_score: float
    match_type: str   # "keyword" | "semantic" | "hybrid" | "deep"


class SearchResponse(BaseModel):
    """Paginated search response returned by search endpoints."""

    results: list[SearchResultItem]
    total: int
    query: str
    mode: str


# ── Annotations ───────────────────────────────────────────────────────────────

class AnnotationRequest(BaseModel):
    """Request body for POST /papers/{paper_id}/annotate."""

    paper_id: UUID
    highlighted_text: str
    note: str | None = None


# ── Deep Search ───────────────────────────────────────────────────────────────

class DeepSearchRequest(BaseModel):
    """Request body for POST /search/deep and POST /search/deep-bg.

    Triggers an LLM-assisted search pipeline: query validation, rewriting,
    semantic + keyword + graph-concept retrieval, and LLM re-ranking.
    """

    query: str = Field(min_length=3, max_length=500, description="Natural-language research query")
    namespace_keys: list[str] | None = Field(
        default=None,
        description="Scope to specific namespaces (e.g. ['cs.AI', 'cs.LG']). "
                    "Omit to search all indexed papers.",
    )
    limit: int = Field(default=20, ge=1, le=50, description="Maximum results to return")
    include_arxiv_mcp: bool = Field(
        default=True,
        description="When useful, fetch fresh arXiv results through MCP and import non-duplicates into the feed.",
    )
    arxiv_max_results: int = Field(default=8, ge=0, le=25)


class DeepSearchJobResponse(BaseModel):
    """Response for both inline and background deep-search requests.

    For inline requests (POST /search/deep) the ``status`` will always be
    ``"done"`` and ``results`` will be populated.  For background requests
    (POST /search/deep-bg) the initial response has ``status="pending"`` and
    callers should poll GET /search/deep/status/{job_id}.
    """

    job_id: str
    status: str          # "pending" | "done" | "failed"
    query: str
    rewritten_query: str | None = None
    results: list[SearchResultItem] | None = None
    error: str | None = None
    cached: bool = False
    imported_count: int = 0


# ── Research Assistant ────────────────────────────────────────────────────────

class AssistantSessionCreateRequest(BaseModel):
    """Request body for creating a Research Assistant session."""

    title: str | None = Field(default=None, max_length=240)
    namespace_key: str = Field(min_length=1, max_length=120)
    topic_keys: list[str] = Field(default_factory=list, max_length=32)


class AssistantMessageRequest(BaseModel):
    """Request body for submitting a message to a Research Assistant session.

    Bounds: ``namespace_key`` 1–120 chars; up to 32 topic keys per turn; up to
    16 attachment refs. These caps mirror the implicit DB column lengths so an
    overflow trips validation before reaching the orchestrator.
    """

    content: str = Field(min_length=1, max_length=6000)
    namespace_key: str = Field(min_length=1, max_length=120)
    topic_keys: list[str] = Field(default_factory=list, max_length=32)
    attachments: list[dict] = Field(default_factory=list, max_length=16)


class AssistantBranchRequest(BaseModel):
    """Request body for branching an existing Research Assistant session."""

    from_message_id: UUID | None = None
    title: str | None = Field(default=None, max_length=240)


class AssistantMessageResponse(BaseModel):
    """Persisted assistant workspace message."""

    id: UUID
    session_id: UUID
    role: str
    content: str
    message_type: str
    citations: list[str]
    artifact_refs: list[dict]
    payload: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class AssistantTaskResponse(BaseModel):
    """Persisted assistant background task."""

    id: UUID
    job_id: str
    session_id: UUID
    assistant_message_id: UUID | None
    task_type: str
    title: str
    namespace_key: str
    status: str
    progress: dict
    result: dict
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class AssistantSessionResponse(BaseModel):
    """Persistent Research Assistant session summary/detail."""

    id: UUID
    title: str
    namespace_key: str
    topic_keys: list[str]
    parent_session_id: UUID | None
    branch_from_message_id: UUID | None
    orientation: str
    expertise_level: str
    summary: str | None
    state: dict
    status: str
    created_at: datetime
    updated_at: datetime
    messages: list[AssistantMessageResponse] = []
    tasks: list[AssistantTaskResponse] = []

    model_config = {"from_attributes": True}


class AssistantSubmitResponse(BaseModel):
    """Response after queuing an assistant orchestration turn."""

    session: AssistantSessionResponse
    user_message: AssistantMessageResponse
    assistant_message: AssistantMessageResponse
    task: AssistantTaskResponse


class AssistantStepResponse(BaseModel):
    """A single tool invocation inside an assistant turn (reasoning-tree node)."""

    id: UUID
    session_id: UUID
    parent_message_id: UUID
    parent_step_id: UUID | None
    job_id: str
    step_index: int
    tool_name: str
    title: str
    status: str
    input_params: dict
    output: dict
    progress: dict
    cost: dict
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AssistantArtifactResponse(BaseModel):
    """Generated output anchored in an assistant session."""

    id: UUID
    session_id: UUID
    user_id: UUID
    producing_step_id: UUID | None
    producing_message_id: UUID | None
    kind: str
    ref_id: str
    title: str
    href: str | None
    preview: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class AssistantToolDescriptor(BaseModel):
    """Schema-only view of a registered assistant tool."""

    name: str
    summary: str
    cost_class: str
    side_effects: bool
    cancellable: bool
    streamable: bool
    input_schema: dict
    output_schema: dict


class AssistantAttachmentCreateRequest(BaseModel):
    """Create a session-scoped attachment (note / URL / paper-ref / image / pdf)."""

    kind: str = Field(min_length=1, max_length=40)  # note | url | paper_ref | pdf | image
    label: str = Field(default="", max_length=240)
    content: str | None = None
    url: str | None = None
    paper_id: UUID | None = None
    metadata: dict = Field(default_factory=dict)


class AssistantAttachmentResponse(BaseModel):
    """Persisted attachment row attached to an assistant session.

    The model attribute is ``metadata_`` (trailing underscore — ``metadata``
    is reserved by SQLAlchemy's declarative base). The Pydantic alias keeps
    the JSON contract clean (``metadata``) while still reading from the model.
    """

    id: UUID
    session_id: UUID
    user_id: UUID
    message_id: UUID | None
    kind: str
    label: str
    content: str | None
    url: str | None
    paper_id: UUID | None
    metadata: dict = Field(default_factory=dict, validation_alias="metadata_",
                           serialization_alias="metadata")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class AssistantSessionRenameRequest(BaseModel):
    """Body for PATCH /assistant/sessions/{id}/title."""

    title: str = Field(min_length=1, max_length=240)


class ArxivSearchRequest(BaseModel):
    """Search arXiv through MCP/RSS fallback."""

    query: str = Field(min_length=2, max_length=500)
    namespace_keys: list[str] = []
    max_results: int = Field(default=10, ge=1, le=50)


class ArxivImportRequest(BaseModel):
    """Import selected arXiv papers into the feed.

    Bounds protect against memory-pressure attacks: ``papers`` is capped at
    100 entries; each is a free-form dict but the service layer rejects ones
    missing required fields. ``namespace_key`` must be non-empty.
    """

    namespace_key: str = Field(min_length=1, max_length=120)
    papers: list[dict] = Field(min_length=1, max_length=100)
