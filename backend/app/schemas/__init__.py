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
    title: str
    abstract: str | None = None
    authors: list[str]
    namespace_key: str
    source_url: str
    pdf_url: str | None
    novelty_score: float
    relevance_score: float
    is_breakthrough: bool
    key_concepts: list[str] | None = None
    methods_used: list[str] | None = None
    implications: str | None = None
    published_at: datetime | None = None
    ingested_at: datetime | None = None
    tldr: str | None = None
    search_score: float
    match_type: str   # "keyword" | "semantic" | "hybrid"


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
