"""GraphRepository — KnowledgeNode and KnowledgeEdge operations."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.graph import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType, SourceMapping


class GraphRepository:
    """Data-access layer for ``KnowledgeNode``, ``KnowledgeEdge``, and ``SourceMapping`` tables.

    Provides idempotent node/edge creation and helpers for reading graph
    structure. All methods are async and require an active ``AsyncSession``.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the repository with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for all queries.
        """
        self._db = db

    async def get_or_create_node(
        self,
        node_type: NodeType,
        label: str,
        namespace_key: str | None = None,
        paper_id: UUID | None = None,
    ) -> KnowledgeNode:
        """Return an existing node matching (label, node_type, namespace_key) or create it.

        The lookup is performed before any insert so the operation is
        idempotent — calling this multiple times with the same arguments
        always returns the same node.

        Args:
            node_type: The ``NodeType`` enum value (e.g. ``topic``,
                ``subtopic``, ``paper``, ``concept``, ``method``).
            label: Human-readable label for the node.
            namespace_key: Optional arXiv-style namespace key. ``None`` for
                TOPIC nodes that span all namespaces.
            paper_id: Optional UUID of the associated ``Paper`` row (only
                relevant for ``NodeType.paper`` nodes).

        Returns:
            The existing or newly created ``KnowledgeNode`` ORM object.
        """
        result = await self._db.execute(
            select(KnowledgeNode).where(
                KnowledgeNode.label == label,
                KnowledgeNode.node_type == node_type,
                KnowledgeNode.namespace_key == namespace_key,
            )
        )
        # Use .first() instead of scalar_one_or_none() because PostgreSQL unique
        # constraints don't treat NULL=NULL as equal, so namespace_key=None nodes
        # can accumulate duplicates from concurrent builds. .first() is resilient.
        node = result.scalars().first()
        if node is None:
            # Use a SAVEPOINT so a concurrent insert (race condition) only rolls back
            # this nested transaction, not the entire outer session transaction.
            from sqlalchemy.exc import IntegrityError as _IntegrityError
            try:
                async with self._db.begin_nested():
                    node = KnowledgeNode(
                        node_type=node_type,
                        label=label,
                        namespace_key=namespace_key,
                        paper_id=paper_id,
                    )
                    self._db.add(node)
            except _IntegrityError:
                # Another concurrent task created this node — fetch it
                result2 = await self._db.execute(
                    select(KnowledgeNode).where(
                        KnowledgeNode.label == label,
                        KnowledgeNode.node_type == node_type,
                        KnowledgeNode.namespace_key == namespace_key,
                    )
                )
                node = result2.scalars().first()
        return node

    async def create_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: EdgeType,
        weight: float = 1.0,
        cross_namespace: bool = False,
    ) -> KnowledgeEdge:
        """Create a directed edge between two nodes, skipping duplicates.

        The operation is idempotent: if an edge with the same
        ``(source_id, target_id, edge_type)`` triple already exists it is
        returned unchanged without creating a second row.

        Args:
            source_id: UUID of the source ``KnowledgeNode``.
            target_id: UUID of the target ``KnowledgeNode``.
            edge_type: The ``EdgeType`` enum value describing the relationship.
            weight: Edge weight used for ranking/traversal. Defaults to ``1.0``.
            cross_namespace: Set to ``True`` when the edge spans two different
                namespaces. Defaults to ``False``.

        Returns:
            The existing or newly created ``KnowledgeEdge`` ORM object.
        """
        # Idempotent — skip if edge already exists
        result = await self._db.execute(
            select(KnowledgeEdge).where(
                KnowledgeEdge.source_id == source_id,
                KnowledgeEdge.target_id == target_id,
                KnowledgeEdge.edge_type == edge_type,
            )
        )
        edge = result.scalar_one_or_none()
        if edge is None:
            edge = KnowledgeEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                cross_namespace=cross_namespace,
            )
            self._db.add(edge)
            await self._db.flush()
        return edge

    async def get_subgraph(
        self, namespace_key: str | None, depth: int = 2
    ) -> tuple[list[KnowledgeNode], list[KnowledgeEdge]]:
        """Return all nodes and their outgoing edges, optionally scoped to a namespace.

        TOPIC nodes have ``namespace_key=None`` (they span namespaces) so they
        are always included regardless of the namespace filter — otherwise the
        hierarchy root would never appear in the result.

        Args:
            namespace_key: Restrict to nodes belonging to this arXiv-style
                namespace key. Pass ``None`` to return nodes from all namespaces.
            depth: Reserved for future depth-limited traversal. Currently unused
                but accepted for forward-compatible call-sites. Defaults to ``2``.

        Returns:
            A two-tuple ``(nodes, edges)`` where ``nodes`` is the list of
            matching ``KnowledgeNode`` objects and ``edges`` is the list of
            ``KnowledgeEdge`` objects whose source node is in that node set.
        """
        from sqlalchemy import or_
        q = select(KnowledgeNode)
        if namespace_key:
            q = q.where(
                or_(
                    KnowledgeNode.namespace_key == namespace_key,
                    KnowledgeNode.namespace_key.is_(None),  # TOPIC nodes
                )
            )
        nodes_result = await self._db.execute(q)
        nodes = list(nodes_result.scalars())
        node_ids = {n.id for n in nodes}
        if not node_ids:
            return nodes, []

        edges_result = await self._db.execute(
            select(KnowledgeEdge).where(
                KnowledgeEdge.source_id.in_(node_ids)
            )
        )
        edges = list(edges_result.scalars())
        return nodes, edges

    async def expand_node(
        self, node_id: UUID, depth: int = 1
    ) -> tuple[list[KnowledgeNode], list[KnowledgeEdge]]:
        """Return the immediate out-neighbors of a single node.

        Fetches all ``KnowledgeEdge`` rows where ``source_id`` equals
        ``node_id``, then loads the corresponding target ``KnowledgeNode``
        objects.

        Args:
            node_id: UUID of the ``KnowledgeNode`` to expand.
            depth: Reserved for future multi-hop traversal. Currently only
                one hop is performed regardless of this value. Defaults to ``1``.

        Returns:
            A two-tuple ``(neighbor_nodes, outgoing_edges)`` — both lists may
            be empty if the node has no outgoing edges.
        """
        edges_result = await self._db.execute(
            select(KnowledgeEdge).where(KnowledgeEdge.source_id == node_id)
        )
        edges = list(edges_result.scalars())
        neighbor_ids = {e.target_id for e in edges}

        nodes_result = await self._db.execute(
            select(KnowledgeNode).where(KnowledgeNode.id.in_(neighbor_ids))
        )
        return list(nodes_result.scalars()), edges

    async def get_source_mappings(self, namespace_key: str) -> list[SourceMapping]:
        """Return all ``SourceMapping`` rows for a specific namespace.

        Args:
            namespace_key: The arXiv-style namespace key to filter on.

        Returns:
            A list of ``SourceMapping`` ORM objects for the namespace.
        """
        result = await self._db.execute(
            select(SourceMapping).where(SourceMapping.namespace_key == namespace_key)
        )
        return list(result.scalars())

    async def get_all_source_mappings(self) -> list[SourceMapping]:
        """Return every ``SourceMapping`` row in the database.

        Used by the scheduler to determine which namespaces to ingest.

        Returns:
            A list of all ``SourceMapping`` ORM objects.
        """
        result = await self._db.execute(select(SourceMapping))
        return list(result.scalars())
