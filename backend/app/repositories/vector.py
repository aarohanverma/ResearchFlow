"""VectorRepository — pgvector similarity search with embedding-space safety.

Every query filters on (embedding_dim, embedding_provider) to prevent
cross-space contamination. This is enforced at the SQL level — never bypassed.
"""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def _vec_str(v: list[float]) -> str:
    """Serialize a Python float list to the pgvector literal format asyncpg accepts."""
    return f"[{','.join(str(x) for x in v)}]"


class VectorRepository:
    """pgvector similarity-search repository with embedding-space safety guards.

    Every query filters on ``(embedding_dim, embedding_provider)`` to prevent
    cross-space contamination — embeddings from different models or dimensions
    are never mixed in the same similarity computation.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the repository with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for all raw SQL queries.
        """
        self._db = db

    async def similarity_search(
        self,
        query_vector: list[float],
        *,
        namespace_key: str | None = None,
        subject_prefix: str | None = None,
        top_k: int = 8,
        score_threshold: float = 0.7,
        embedding_dim: int = 768,
        embedding_provider: str = "gemini",
    ) -> list[dict]:
        """Cosine similarity search — scoped by namespace (topic → subject → global).

        Returns dicts with: chunk_id, paper_id, title, namespace_key, similarity, content.
        """
        # Build WHERE clause for scope
        scope_filter = ""
        params: dict = {
            "vec": _vec_str(query_vector),
            "top_k": top_k,
            "threshold": score_threshold,
            "dim": embedding_dim,
            "provider": embedding_provider,
        }

        if namespace_key:
            scope_filter = "AND p.namespace_key = :ns"
            params["ns"] = namespace_key
        elif subject_prefix:
            scope_filter = "AND p.namespace_key LIKE :prefix"
            params["prefix"] = f"{subject_prefix}.%"

        sql = text(f"""
            SELECT
                pc.id          AS chunk_id,
                p.id           AS paper_id,
                p.title,
                p.namespace_key,
                pc.content,
                1 - (pc.embedding <=> CAST(:vec AS vector)) AS similarity
            FROM paper_chunks pc
            JOIN papers p ON pc.paper_id = p.id
            WHERE pc.embedding IS NOT NULL
              AND pc.embedding_dim = :dim
              AND pc.embedding_provider = :provider
              AND 1 - (pc.embedding <=> CAST(:vec AS vector)) >= :threshold
              {scope_filter}
            ORDER BY pc.embedding <=> CAST(:vec AS vector)
            LIMIT :top_k
        """)

        result = await self._db.execute(sql, params)
        return [dict(row._mapping) for row in result.fetchall()]

    async def find_similar_chunks(
        self,
        chunk_ids: list[UUID],
        query_vector: list[float],
        top_k: int = 5,
        embedding_dim: int = 768,
        embedding_provider: str = "gemini",
    ) -> list[dict]:
        """Given candidate chunk IDs, return the top-k by similarity."""
        if not chunk_ids:
            return []
        ids_str = ", ".join(f"'{cid}'" for cid in chunk_ids)
        sql = text(f"""
            SELECT
                pc.id AS chunk_id,
                p.id  AS paper_id,
                p.title,
                pc.content,
                1 - (pc.embedding <=> CAST(:vec AS vector)) AS similarity
            FROM paper_chunks pc
            JOIN papers p ON pc.paper_id = p.id
            WHERE pc.id IN ({ids_str})
              AND pc.embedding IS NOT NULL
              AND pc.embedding_dim = :dim
              AND pc.embedding_provider = :provider
            ORDER BY pc.embedding <=> CAST(:vec AS vector)
            LIMIT :top_k
        """)
        result = await self._db.execute(sql, {
            "vec": _vec_str(query_vector),
            "dim": embedding_dim,
            "provider": embedding_provider,
            "top_k": top_k,
        })
        return [dict(row._mapping) for row in result.fetchall()]

    async def cross_namespace_similar_nodes(
        self,
        node_embedding: list[float],
        *,
        exclude_namespace: str,
        threshold: float = 0.85,
        top_k: int = 10,
        embedding_dim: int = 768,
        embedding_provider: str = "gemini",
    ) -> list[dict]:
        """Find concept nodes in other namespaces with high cosine similarity.
        Used weekly to build cross-namespace related_to edges.
        """
        sql = text("""
            SELECT
                pc.id          AS chunk_id,
                p.id           AS paper_id,
                p.namespace_key,
                p.title,
                1 - (pc.embedding <=> CAST(:vec AS vector)) AS similarity
            FROM paper_chunks pc
            JOIN papers p ON pc.paper_id = p.id
            WHERE pc.embedding IS NOT NULL
              AND pc.embedding_dim = :dim
              AND pc.embedding_provider = :provider
              AND p.namespace_key != :excl_ns
              AND 1 - (pc.embedding <=> CAST(:vec AS vector)) >= :threshold
            ORDER BY pc.embedding <=> CAST(:vec AS vector)
            LIMIT :top_k
        """)
        result = await self._db.execute(sql, {
            "vec": _vec_str(node_embedding),
            "dim": embedding_dim,
            "provider": embedding_provider,
            "excl_ns": exclude_namespace,
            "threshold": threshold,
            "top_k": top_k,
        })
        return [dict(row._mapping) for row in result.fetchall()]
