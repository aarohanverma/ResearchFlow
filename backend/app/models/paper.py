"""Paper, PaperChunk, Summary, Bookmark, PaperOfDay, QueryLog models."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base
from app.models.user import EmbeddingProvider, ExpertiseLevel  # noqa: F401 — shared enums


class Paper(Base):
    """ORM model for an ingested research paper.

    Each row corresponds to a unique (external_id, namespace_key) pair.
    Enrichment fields (key_concepts, methods_used, etc.) are populated by
    the ``enrich_papers`` workflow node. Study flags are set during
    ``StudyWorkflow``.
    """

    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)  # arXiv ID
    namespace_key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # e.g. cs.AI

    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Enrichment (set by enrich_papers node)
    key_concepts: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    methods_used: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    implications: Mapped[str | None] = mapped_column(Text, nullable=True)
    novelty_score: Mapped[float] = mapped_column(Float, default=0.0)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Study flags — set during StudyWorkflow
    has_algorithm: Mapped[bool] = mapped_column(Boolean, default=False)
    has_architecture: Mapped[bool] = mapped_column(Boolean, default=False)
    has_dataflow: Mapped[bool] = mapped_column(Boolean, default=False)
    needs_rich_diagram: Mapped[bool] = mapped_column(Boolean, default=False)

    # PDF parse status
    pdf_parsed: Mapped[bool] = mapped_column(Boolean, default=False)
    pdf_blob_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Parser metadata — populated by ingestion / study workflows so downstream
    # generation can include parser provenance in cache keys and audit trails.
    parser_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    parser_fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    parse_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parser_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Breakthrough flag
    is_breakthrough: Mapped[bool] = mapped_column(Boolean, default=False)

    # AI-generated one-liner (cached, generated lazily via Haiku)
    tldr: Mapped[str | None] = mapped_column(Text, nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("external_id", "namespace_key", name="uq_paper_external_ns"),
    )

    chunks: Mapped[list["PaperChunk"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    summaries: Mapped[list["Summary"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    bookmarks: Mapped[list["Bookmark"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    citations_from: Mapped[list["PaperCitation"]] = relationship(
        back_populates="source_paper",
        foreign_keys="PaperCitation.source_paper_id",
        cascade="all, delete-orphan",
    )
    citations_to: Mapped[list["PaperCitation"]] = relationship(
        back_populates="cited_paper",
        foreign_keys="PaperCitation.cited_paper_id",
    )


class PaperChunk(Base):
    """Embedding unit — one per paper initially (abstract), more after Study."""

    __tablename__ = "paper_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )

    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    section_type: Mapped[str] = mapped_column(String(50), default="abstract")
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Vector — dimension kept flexible; index filtered by dim+provider
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    embedding_dim: Mapped[int] = mapped_column(Integer, default=768)
    embedding_provider: Mapped[EmbeddingProvider] = mapped_column(
        Enum(EmbeddingProvider), default=EmbeddingProvider.gemini
    )

    # Figure chunks — optional image bytes stored in blob, caption here
    is_figure: Mapped[bool] = mapped_column(Boolean, default=False)
    figure_caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    figure_blob_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["Paper"] = relationship(back_populates="chunks")


class Summary(Base):
    """Cached Study output — max 3 rows per paper (one per expertise level).
    Version-bumped when prompt_hash or model_used changes."""

    __tablename__ = "summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    expertise_level: Mapped[ExpertiseLevel] = mapped_column(Enum(ExpertiseLevel))

    # Content sections stored as structured JSON
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)

    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)

    diagrams: Mapped[list[dict]] = mapped_column(JSONB, default=list)  # DiagramSpec objects
    has_code: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("paper_id", "expertise_level", name="uq_summary_paper_level"),
    )

    paper: Mapped["Paper"] = relationship(back_populates="summaries")


class BookmarkFolder(Base):
    """Named folder for organizing bookmarks. Scopes RAG chat and Genie."""

    __tablename__ = "bookmark_folders"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    color: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_folder_name"),)

    members: Mapped[list["BookmarkFolderMember"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan"
    )


class Bookmark(Base):
    """ORM model representing a user's saved (bookmarked) paper.

    The (user_id, paper_id) pair is unique — a user can bookmark a paper
    only once. Optional ``note`` stores a free-text annotation. Bookmarks
    can belong to multiple ``BookmarkFolder`` rows via the junction table.
    """

    __tablename__ = "bookmarks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("user_id", "paper_id", name="uq_bookmark"),)

    user: Mapped["User"] = relationship(back_populates="bookmarks")  # noqa: F821
    paper: Mapped["Paper"] = relationship(back_populates="bookmarks")
    folder_members: Mapped[list["BookmarkFolderMember"]] = relationship(
        back_populates="bookmark", cascade="all, delete-orphan"
    )


class BookmarkFolderMember(Base):
    """Junction table: one row per (bookmark, folder) pairing (many-to-many)."""

    __tablename__ = "bookmark_folder_members"

    bookmark_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("bookmarks.id", ondelete="CASCADE"), primary_key=True
    )
    folder_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("bookmark_folders.id", ondelete="CASCADE"), primary_key=True, index=True
    )

    bookmark: Mapped["Bookmark"] = relationship(back_populates="folder_members")
    folder: Mapped["BookmarkFolder"] = relationship(back_populates="members")


class PaperOfDay(Base):
    """ORM model for the daily featured paper per namespace.

    At most one row per (namespace_key, date) pair. The ``score`` field
    reflects the combined novelty + relevance score used to select the
    paper. ``is_breakthrough`` is copied from the source paper row.
    """

    __tablename__ = "paper_of_day"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_key: Mapped[str] = mapped_column(String(100), nullable=False)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE")
    )
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    is_breakthrough: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("namespace_key", "date", name="uq_potd"),)

    paper: Mapped["Paper"] = relationship()


class PaperCitation(Base):
    """ORM model for a directed citation link between two papers.

    ``source_paper_id`` cites ``cited_paper_id``. The ``confidence``
    field stores the extraction confidence (defaults to 1.0 for
    deterministic arXiv-ID matches, lower for heuristic matches).
    """

    __tablename__ = "paper_citations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    cited_paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    source_paper: Mapped["Paper"] = relationship(
        back_populates="citations_from", foreign_keys=[source_paper_id]
    )
    cited_paper: Mapped["Paper"] = relationship(
        back_populates="citations_to", foreign_keys=[cited_paper_id]
    )


class QueryLog(Base):
    """RAG query log — drives interest profile updates."""

    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    namespace_key: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_query: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    retrieved_paper_ids: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FeedFeedback(Base):
    """Lightweight feed signals: like / dismiss / more-like-this."""

    __tablename__ = "feed_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    signal: Mapped[str] = mapped_column(String(30), nullable=False)  # like | dismiss | more_like_this
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
