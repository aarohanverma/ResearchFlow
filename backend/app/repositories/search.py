"""SearchRepository — hybrid search combining keyword (PostgreSQL FTS) + semantic (pgvector).

Fusion strategy: Reciprocal Rank Fusion (RRF)
  score(paper) = Σ_i  w_i / (k + rank_i)     k=60 (standard default)

Keyword path: to_tsvector + plainto_tsquery over title, tldr, abstract,
              key_concepts, methods_used — ranked by ts_rank_cd.
Semantic path: cosine similarity on paper_chunks embeddings (requires embedded query).
              Uses ANN index via ORDER BY distance LIMIT to get top candidates,
              then groups by paper_id to keep the best-scoring chunk per paper.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

_RRF_K = 60            # standard RRF constant — tunable
_MAX_KW_RESULTS = 50
_MAX_SEM_RESULTS = 50  # distinct papers returned from semantic path
_SEM_INNER_LIMIT = 200 # ANN candidate pool before dedup — uses IVFFlat index
_MIN_SEM_SCORE = 0.20  # discard chunks with very low cosine similarity
_KW_WEIGHT  = 0.8      # keyword is fallback; hybrid wins when both fire
_SEM_WEIGHT = 1.2      # semantic preferred — captures conceptual matches

# Columns that require the tldr column to exist
_paper_cols = """
                p.id            AS paper_id,
                p.external_id,
                p.title,
                p.abstract,
                p.tldr,
                p.authors,
                p.namespace_key,
                p.source_url,
                p.pdf_url,
                p.novelty_score,
                p.relevance_score,
                p.is_breakthrough,
                p.key_concepts,
                p.methods_used,
                p.implications,
                p.published_at,
                p.ingested_at"""

# Fallback cols — no tldr (safe before migration runs)
_paper_cols_basic = """
                p.id            AS paper_id,
                p.external_id,
                p.title,
                p.abstract,
                NULL::text      AS tldr,
                p.authors,
                p.namespace_key,
                p.source_url,
                p.pdf_url,
                p.novelty_score,
                p.relevance_score,
                p.is_breakthrough,
                p.key_concepts,
                p.methods_used,
                p.implications,
                p.published_at,
                p.ingested_at"""


def _vec_str(v: list[float]) -> str:
    """Serialize a Python float list to the pgvector literal format asyncpg accepts."""
    return f"[{','.join(str(x) for x in v)}]"


class SearchRepository:
    """Hybrid search repository combining PostgreSQL FTS and pgvector similarity.

    Fuses keyword and semantic results using Reciprocal Rank Fusion (RRF).
    Semantic results act purely as a re-ranking signal — they never surface
    papers that did not also match the keyword path.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the repository with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for all raw SQL queries.
        """
        self._db = db

    # ── Public ─────────────────────────────────────────────────────────────────

    async def hybrid_search(
        self,
        query: str,
        *,
        namespace_key: str | None = None,
        namespace_keys: list[str] | None = None,
        query_vector: list[float] | None = None,
        limit: int = 20,
        embedding_dim: int = 768,
        embedding_provider: str = "gemini",
    ) -> list[dict]:
        """Run keyword + (optionally) semantic search and fuse results via RRF.

        ``namespace_keys`` (list) takes precedence over ``namespace_key`` (single).
        When both are ``None`` the search spans ALL indexed papers.

        Keyword path covers title, tldr, abstract, key_concepts, and
        methods_used for improved method/concept recall.  Semantic path uses an
        ANN-friendly two-step approach: LIMIT on the raw vector scan (uses
        IVFFlat index), then GROUP BY paper_id for deduplication, then JOIN to
        papers, with a minimum similarity floor of 0.20.

        Args:
            query: Raw search query string.
            namespace_key: Single namespace filter (ignored when ``namespace_keys`` is set).
            namespace_keys: List of namespace keys; overrides ``namespace_key`` when provided.
            query_vector: Pre-computed query embedding.  When ``None`` the
                search falls back to keyword-only.
            limit: Maximum results to return. Defaults to 20.
            embedding_dim: Dimensionality of stored embeddings.  Defaults to 768.
            embedding_provider: Provider identifier for embedding-space safety
                guard.  Defaults to ``"gemini"``.

        Returns:
            List of dicts with keys: paper_id, title, abstract, authors,
            namespace_key, source_url, pdf_url, novelty_score, relevance_score,
            is_breakthrough, tldr, search_score, match_type
            (``"hybrid"``, ``"keyword"``, or ``"deep"``).
        """
        # Resolve effective namespace filter
        ns_list = namespace_keys or ([namespace_key] if namespace_key else None)

        kw_results = await self._keyword_search(query, namespace_keys=ns_list)

        sem_results: list[dict] = []
        if query_vector:
            sem_results = await self._semantic_search(
                query_vector,
                namespace_keys=ns_list,
                embedding_dim=embedding_dim,
                embedding_provider=embedding_provider,
            )

        fused = self._rrf_fuse(kw_results, sem_results)
        return fused[:limit]

    # ── Keyword Search ─────────────────────────────────────────────────────────

    @staticmethod
    def _ns_filter(ns_list: list[str] | None, params: dict) -> str:
        """Build a namespace WHERE clause and inject params. Returns SQL fragment."""
        if not ns_list:
            return ""
        if len(ns_list) == 1:
            params["ns0"] = ns_list[0]
            return "AND p.namespace_key = :ns0"
        # Multiple — use positional params ns0…nsN
        placeholders = ", ".join(f":ns{i}" for i in range(len(ns_list)))
        for i, ns in enumerate(ns_list):
            params[f"ns{i}"] = ns
        return f"AND p.namespace_key IN ({placeholders})"

    async def _keyword_search(
        self,
        query: str,
        *,
        namespace_keys: list[str] | None = None,
    ) -> list[dict]:
        """Full-text search: english FTS → ILIKE with tldr → ILIKE without tldr."""
        params: dict = {"q": query, "limit": _MAX_KW_RESULTS}
        ns_filter = self._ns_filter(namespace_keys, params)

        # Primary: english FTS (handles stemming — "learning" matches "learn")
        # Includes key_concepts and methods_used for richer method/concept recall.
        fts_sql = text(f"""
            SELECT {_paper_cols},
                ts_rank_cd(
                    to_tsvector('english',
                        COALESCE(p.title, '') || ' ' ||
                        COALESCE(p.tldr, '') || ' ' ||
                        COALESCE(p.abstract, '') || ' ' ||
                        COALESCE(array_to_string(p.key_concepts, ' '), '') || ' ' ||
                        COALESCE(array_to_string(p.methods_used, ' '), '')
                    ),
                    plainto_tsquery('english', :q)
                ) AS kw_score
            FROM papers p
            WHERE to_tsvector('english',
                      COALESCE(p.title, '') || ' ' ||
                      COALESCE(p.tldr, '') || ' ' ||
                      COALESCE(p.abstract, '') || ' ' ||
                      COALESCE(array_to_string(p.key_concepts, ' '), '') || ' ' ||
                      COALESCE(array_to_string(p.methods_used, ' '), '')
                  ) @@ plainto_tsquery('english', :q)
              {ns_filter}
            ORDER BY kw_score DESC
            LIMIT :limit
        """)

        try:
            result = await self._db.execute(fts_sql, params)
            rows = result.fetchall()
        except Exception as exc:
            log.warning("fts search failed: %s", exc)
            rows = []

        if rows:
            return [dict(row._mapping) for row in rows]

        # ILIKE fallback — catches acronyms, model names, short terms FTS misses
        like_params: dict = {"q": f"%{query}%", "limit": _MAX_KW_RESULTS}
        like_ns = self._ns_filter(namespace_keys, like_params)

        like_sql = text(f"""
            SELECT {_paper_cols}, 0.1::float AS kw_score
            FROM papers p
            WHERE (
                COALESCE(p.title,    '') ILIKE :q OR
                COALESCE(p.tldr,     '') ILIKE :q OR
                COALESCE(p.abstract, '') ILIKE :q OR
                COALESCE(array_to_string(p.key_concepts, ' '), '') ILIKE :q OR
                COALESCE(array_to_string(p.methods_used,  ' '), '') ILIKE :q
            )
            {like_ns}
            ORDER BY p.novelty_score DESC
            LIMIT :limit
        """)

        try:
            result = await self._db.execute(like_sql, like_params)
            rows = result.fetchall()
        except Exception as exc:
            log.warning("ilike fallback failed: %s", exc)
            rows = []

        if rows:
            return [dict(row._mapping) for row in rows]

        # Last-resort: title + abstract only — safe before tldr migration runs
        basic_params: dict = {"q": f"%{query}%", "limit": _MAX_KW_RESULTS}
        basic_ns = self._ns_filter(namespace_keys, basic_params)
        basic_sql = text(f"""
            SELECT {_paper_cols_basic}, 0.05::float AS kw_score
            FROM papers p
            WHERE (
                COALESCE(p.title,    '') ILIKE :q OR
                COALESCE(p.abstract, '') ILIKE :q
            )
            {basic_ns}
            ORDER BY p.novelty_score DESC
            LIMIT :limit
        """)

        try:
            result = await self._db.execute(basic_sql, basic_params)
            rows = result.fetchall()
        except Exception as exc:
            log.warning("basic ilike fallback failed: %s", exc)
            return []

        return [dict(row._mapping) for row in rows]

    # ── Semantic Search ────────────────────────────────────────────────────────

    async def _semantic_search(
        self,
        query_vector: list[float],
        *,
        namespace_keys: list[str] | None = None,
        embedding_dim: int = 768,
        embedding_provider: str = "gemini",
    ) -> list[dict]:
        """Vector similarity search — returns distinct papers (best chunk per paper).

        Uses a two-step approach that leverages the IVFFlat ANN index:
          1. ORDER BY distance LIMIT inner_limit  →  uses the ANN index (fast)
          2. GROUP BY paper_id keeping max score   →  one result per paper
          3. Filter below minimum similarity floor →  remove noise
          4. JOIN papers, apply namespace filter, ORDER BY score DESC, LIMIT
        """
        vec_literal = _vec_str(query_vector)
        params: dict = {
            "vec": vec_literal,
            "dim": embedding_dim,
            "provider": embedding_provider,
            "inner_limit": _SEM_INNER_LIMIT,
            "outer_limit": _MAX_SEM_RESULTS,
            "min_score": _MIN_SEM_SCORE,
        }
        ns_clause = self._ns_filter(namespace_keys, params)
        ns_where = f"AND {ns_clause[4:]}" if ns_clause else ""  # strip leading "AND "

        sql = text(f"""
            SELECT
                p.id            AS paper_id,
                p.external_id,
                p.title,
                p.abstract,
                p.tldr,
                p.authors,
                p.namespace_key,
                p.source_url,
                p.pdf_url,
                p.novelty_score,
                p.relevance_score,
                p.is_breakthrough,
                p.key_concepts,
                p.methods_used,
                p.implications,
                p.published_at,
                p.ingested_at,
                best.sem_score
            FROM (
                SELECT paper_id, MAX(sem_score) AS sem_score
                FROM (
                    SELECT
                        pc.paper_id,
                        1 - (pc.embedding <=> CAST(:vec AS vector)) AS sem_score
                    FROM paper_chunks pc
                    WHERE pc.embedding IS NOT NULL
                      AND pc.embedding_dim = :dim
                      AND pc.embedding_provider = :provider
                    ORDER BY pc.embedding <=> CAST(:vec AS vector)
                    LIMIT :inner_limit
                ) chunks_raw
                WHERE sem_score >= :min_score
                GROUP BY paper_id
            ) best
            JOIN papers p ON p.id = best.paper_id
            WHERE 1=1 {ns_where}
            ORDER BY best.sem_score DESC
            LIMIT :outer_limit
        """)

        try:
            result = await self._db.execute(sql, params)
            rows = result.fetchall()
        except Exception as exc:
            log.warning("semantic search failed — falling back to empty: %s", exc, exc_info=True)
            return []

        return [dict(row._mapping) for row in rows]

    # ── RRF Fusion ─────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        kw_results: list[dict],
        sem_results: list[dict],
    ) -> list[dict]:
        """Reciprocal Rank Fusion — hybrid by default, keyword-only as fallback.

        Strategy:
          - Hybrid: paper appears in BOTH keyword and semantic results (shown, scored highest)
          - Keyword: paper appears in keyword results only (shown)
          - Semantic-only: paper appears only in semantic results (excluded — never shown alone)

        Semantic is used purely as a re-ranking signal, not an independent result source.
        Deduplicates by external_id to remove cross-namespace copies of the same arXiv paper.
        """
        scores: dict[str, float] = {}
        paper_data: dict[str, dict] = {}

        # Semantic scores first (re-ranking signal only — will be dropped if no kw match)
        for rank, row in enumerate(sem_results, start=1):
            pid = str(row["paper_id"])
            scores[pid] = scores.get(pid, 0.0) + _SEM_WEIGHT / (_RRF_K + rank)
            if pid not in paper_data:
                paper_data[pid] = {**row, "match_type": "semantic"}

        # Keyword scores — anchors the result set; promotes semantic entries to "hybrid"
        for rank, row in enumerate(kw_results, start=1):
            pid = str(row["paper_id"])
            scores[pid] = scores.get(pid, 0.0) + _KW_WEIGHT / (_RRF_K + rank)
            if pid not in paper_data:
                paper_data[pid] = {**row, "match_type": "keyword"}
            else:
                paper_data[pid]["match_type"] = "hybrid"

        # Sort by fused RRF score
        sorted_ids = sorted(scores, key=lambda p: scores[p], reverse=True)

        # Emit only papers that appear in keyword results (hybrid or keyword).
        # Semantic-only papers are excluded: semantic re-ranks but never adds new results.
        # Deduplicate by external_id to remove cross-namespace copies.
        seen_external: set[str] = set()
        results = []
        for pid in sorted_ids:
            row = paper_data[pid]
            if row.get("match_type") == "semantic":
                continue  # no keyword match — exclude
            eid = str(row.get("external_id") or "")
            if eid and eid in seen_external:
                continue
            if eid:
                seen_external.add(eid)
            results.append({**row, "search_score": round(scores[pid], 6)})

        return results
