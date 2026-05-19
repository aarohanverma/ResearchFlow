"""User, UserProviderSettings, UserInterestProfile, Annotation models."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey,
    String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class ExpertiseLevel(str, enum.Enum):
    """User-selected expertise level that controls summary depth and vocabulary."""

    newcomer = "newcomer"
    practitioner = "practitioner"
    expert = "expert"


class Orientation(str, enum.Enum):
    """User's primary interest orientation — affects feed scoring weights."""

    research = "research"
    production = "production"
    both = "both"


class LLMProvider(str, enum.Enum):
    """Supported LLM provider identifiers stored in user provider settings."""

    openai = "openai"
    anthropic = "anthropic"
    google = "google"


class EmbeddingProvider(str, enum.Enum):
    """Supported embedding provider identifiers stored in user provider settings."""

    gemini = "gemini"
    openai = "openai"
    voyage = "voyage"


class User(Base):
    """ORM model for a registered user account.

    Stores authentication credentials (hashed password), display preferences,
    expertise/orientation profile, and notification flags. Related rows are
    linked via ``provider_settings``, ``interest_profile``, ``bookmarks``,
    ``annotations``, and ``namespace_subscriptions`` relationships.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False, default="Researcher")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Admin flag — admins access the management panel (users, analytics,
    # graph rebuild) but otherwise share the same routes as regular users.
    # Stored as a column rather than a separate Role enum because today
    # there are only two states (regular vs admin); a Role table would be
    # premature.
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Per-user feature overrides. Empty dict means "inherit global defaults".
    # Each key matches the canonical feature flags listed in
    # ``app.services.feature_flags.DEFAULTS``. Overrides win over global
    # admin-set flags so a single user can be granted access to a feature
    # the rest of the org is gated out of (or vice-versa).
    feature_overrides: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    # Forward-compatible role + tier slot. ``role`` is a string today
    # (``admin``/``member``) — mirrors ``is_admin`` but lets us add more
    # roles (e.g. ``editor``) without a migration. ``tier_slug`` joins
    # to ``tiers.slug`` to inherit a feature_set + quotas; NULL means
    # the user simply uses the global flag values (the only state that
    # exists today). The full resolution chain is documented in
    # ``app.services.feature_flags``.
    role: Mapped[str] = mapped_column(String(40), default="member", nullable=False)
    tier_slug: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    onboarding_complete: Mapped[bool] = mapped_column(Boolean, default=False)

    # Depth profile
    expertise_level: Mapped[ExpertiseLevel] = mapped_column(
        Enum(ExpertiseLevel), default=ExpertiseLevel.practitioner
    )
    orientation: Mapped[Orientation] = mapped_column(
        Enum(Orientation), default=Orientation.both
    )

    # Notification preferences
    notify_potd: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_digest: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_breakthrough: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    provider_settings: Mapped["UserProviderSettings"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    interest_profile: Mapped["UserInterestProfile"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    bookmarks: Mapped[list["Bookmark"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    namespace_subscriptions: Mapped[list["NamespaceSubscription"]] = relationship(  # noqa: F821
        back_populates="user", cascade="all, delete-orphan"
    )


class UserProviderSettings(Base):
    """Per-user runtime-swappable provider/model configuration."""

    __tablename__ = "user_provider_settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )

    llm_provider: Mapped[LLMProvider] = mapped_column(Enum(LLMProvider), default=LLMProvider.openai)
    cheap_model: Mapped[str] = mapped_column(String(100), default="gpt-4o-mini")
    quality_model: Mapped[str] = mapped_column(String(100), default="gpt-5.4-mini")
    reasoning_model: Mapped[str] = mapped_column(String(100), default="gpt-5.4")

    embedding_provider: Mapped[EmbeddingProvider] = mapped_column(
        Enum(EmbeddingProvider), default=EmbeddingProvider.gemini
    )
    embedding_model: Mapped[str] = mapped_column(String(100), default="gemini-embedding-2-preview")
    embedding_dim: Mapped[int] = mapped_column(default=768)

    # Encrypted user-supplied API keys (envelope encryption)
    encrypted_openai_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_anthropic_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_google_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_wolfram_key: Mapped[str | None] = mapped_column(Text, nullable=True)  # Wolfram Alpha App ID

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="provider_settings")


class UserInterestProfile(Base):
    """Soft signals derived from behaviour — never trust blindly."""

    __tablename__ = "user_interest_profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )

    # Lists of subtopic node IDs (KnowledgeNode UUIDs as strings for simplicity)
    hot_subtopics: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    cold_subtopics: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    # concept_affinity: {concept_node_id: weight}
    concept_affinity: Mapped[dict] = mapped_column(JSONB, default=dict)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="interest_profile")


class Annotation(Base):
    """User highlights / notes on paper text — feed into personal_retrieve in RAG."""

    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )

    highlighted_text: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Embedding computed lazily and stored — used in personal_retrieve
    embedding: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="annotations")
    paper: Mapped["Paper"] = relationship()  # noqa: F821
