"""Knowledge Graph nodes and edges. No external graph DB — pure PostgreSQL."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class NodeType(str, enum.Enum):
    """Taxonomy level of a node in the knowledge graph."""

    topic = "TOPIC"
    subtopic = "SUBTOPIC"
    concept = "CONCEPT"
    method = "METHOD"
    paper = "PAPER"


class EdgeType(str, enum.Enum):
    """Directed relationship type between two knowledge graph nodes."""

    has_subtopic = "has_subtopic"
    belongs_to = "belongs_to"
    introduces = "introduces"
    uses_method = "uses_method"
    cites = "cites"
    related_to = "related_to"


class KnowledgeNode(Base):
    """ORM model for a node in the knowledge graph.

    Nodes are arranged in a TOPIC → SUBTOPIC → CONCEPT/METHOD → PAPER
    hierarchy stored entirely in PostgreSQL (no external graph database).
    PAPER nodes carry a ``paper_id`` foreign key back to the ``papers``
    table. The ``user_label`` field allows subtopic nodes to be renamed
    after LLM-driven clustering.
    """

    __tablename__ = "knowledge_nodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_type: Mapped[NodeType] = mapped_column(Enum(NodeType), nullable=False, index=True)

    # Canonical label — used for embedding + display
    label: Mapped[str] = mapped_column(String(500), nullable=False)
    namespace_key: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # For PAPER nodes, links back to the papers table
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=True
    )

    # User-overridden label for Subtopic nodes (from clustering rename)
    user_label: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("label", "node_type", "namespace_key", name="uq_node_label_type_ns"),
    )

    outgoing_edges: Mapped[list["KnowledgeEdge"]] = relationship(
        back_populates="source_node",
        foreign_keys="KnowledgeEdge.source_id",
        cascade="all, delete-orphan",
    )
    incoming_edges: Mapped[list["KnowledgeEdge"]] = relationship(
        back_populates="target_node",
        foreign_keys="KnowledgeEdge.target_id",
    )


class KnowledgeEdge(Base):
    """ORM model for a directed edge in the knowledge graph.

    Connects two ``KnowledgeNode`` rows with a typed relationship. The
    ``weight`` column stores cosine similarity for ``RELATED_TO`` edges
    computed at link-creation time. ``cross_namespace`` is set when source
    and target nodes belong to different namespaces.
    """

    __tablename__ = "knowledge_edges"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_nodes.id", ondelete="CASCADE"), index=True
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_nodes.id", ondelete="CASCADE"), index=True
    )
    edge_type: Mapped[EdgeType] = mapped_column(Enum(EdgeType), nullable=False)

    # Weight for RELATED_TO edges — cosine similarity at link creation time
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    # Cross-namespace edges have this flag — helps UI render them distinctly
    cross_namespace: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "edge_type", name="uq_edge"),
    )

    source_node: Mapped["KnowledgeNode"] = relationship(
        back_populates="outgoing_edges", foreign_keys=[source_id]
    )
    target_node: Mapped["KnowledgeNode"] = relationship(
        back_populates="incoming_edges", foreign_keys=[target_id]
    )


class NamespaceSubscription(Base):
    """Maps a user to the namespaces they subscribed to during onboarding."""

    __tablename__ = "namespace_subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    namespace_key: Mapped[str] = mapped_column(String(100), nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "namespace_key", name="uq_ns_sub"),)

    user: Mapped["User"] = relationship(back_populates="namespace_subscriptions")  # noqa: F821


class SourceMapping(Base):
    """Maps a namespace_key (e.g. cs.AI) to an external source category (cs.AI → arXiv cs.AI)."""

    __tablename__ = "source_mappings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    source_name: Mapped[str] = mapped_column(String(50), nullable=False)      # arxiv_rss | arxiv_mcp
    external_category_key: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. cs.AI

    __table_args__ = (
        UniqueConstraint("namespace_key", "source_name", name="uq_source_mapping"),
    )
