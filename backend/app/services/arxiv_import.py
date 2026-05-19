"""arXiv MCP search and feed import service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.sources.arxiv_mcp import ArXivMcpSource
from app.adapters.sources.base import RawPaper
from app.models.graph import SourceMapping
from app.models.paper import Paper, PaperChunk
from app.repositories.paper import PaperRepository
from app.services.graph import GraphService
from app.services.namespace import NAMESPACE_TO_ARXIV, NamespaceManager

log = logging.getLogger(__name__)


class ArxivImportService:
    """Search arXiv externally and import selected results into ResearchFlow."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def search(
        self,
        query: str,
        *,
        namespace_keys: list[str] | None = None,
        max_results: int = 10,
    ) -> list[dict]:
        source = ArXivMcpSource()
        papers = await source.search(query, max_results=max_results, namespace_keys=namespace_keys or None)
        return [self._raw_to_dict(p) for p in papers]

    async def import_raw_papers(
        self,
        raw_papers: list[RawPaper | dict],
        *,
        namespace_key: str,
        create_embeddings: bool = True,
        update_graph: bool = True,
    ) -> tuple[list[Paper], int]:
        """Import raw arXiv papers into the feed without duplicating rows.

        Returns:
            ``(new_papers, skipped_count)`` where ``skipped_count`` is the
            number of candidate papers already present for this namespace.
        """
        if not raw_papers:
            return [], 0

        await self._ensure_source_mapping(namespace_key)

        normalized: list[dict] = []
        for item in raw_papers:
            data = self._raw_to_dict(item) if isinstance(item, RawPaper) else dict(item)
            external_id = str(data.get("external_id") or data.get("arxiv_id") or "").strip()
            if not external_id:
                continue
            normalized.append({
                "external_id": external_id,
                "namespace_key": namespace_key,
                "title": str(data.get("title") or "").strip() or external_id,
                "authors": list(data.get("authors") or ["Unknown"]),
                "abstract": str(data.get("abstract") or "").strip(),
                "source_url": data.get("source_url") or f"https://arxiv.org/abs/{external_id}",
                "pdf_url": data.get("pdf_url") or f"https://arxiv.org/pdf/{external_id}.pdf",
                "published_at": self._coerce_dt(data.get("published_at")),
                "key_concepts": list(data.get("key_concepts") or []),
                "methods_used": list(data.get("methods_used") or []),
                "implications": data.get("implications"),
                "novelty_score": float(data.get("novelty_score") or 0.55),
                "relevance_score": float(data.get("relevance_score") or 0.55),
                "tldr": data.get("tldr"),
            })

        if not normalized:
            return [], 0

        pairs = [(p["external_id"], p["namespace_key"]) for p in normalized]
        existing = await self._db.execute(
            select(Paper.external_id, Paper.namespace_key).where(
                tuple_(Paper.external_id, Paper.namespace_key).in_(pairs)
            )
        )
        existing_pairs = {(row.external_id, row.namespace_key) for row in existing.fetchall()}
        skipped = sum(1 for p in normalized if (p["external_id"], p["namespace_key"]) in existing_pairs)

        repo = PaperRepository(self._db)
        new_papers = await repo.upsert_papers(normalized)
        await self._db.commit()

        if create_embeddings and new_papers:
            await self._embed_abstracts(new_papers)

        if update_graph and new_papers:
            await self._index_graph(new_papers)

        return new_papers, skipped

    async def import_search_results(
        self,
        query: str,
        *,
        namespace_key: str,
        namespace_keys: list[str] | None = None,
        max_results: int = 8,
    ) -> tuple[list[Paper], int, list[dict]]:
        """Search arXiv externally and import new papers into the active feed.

        ``namespace_keys`` acts as an optional category filter for the Atom API
        — pass it only when the user explicitly scopes to specific arXiv
        categories. When None, the search is cross-arXiv (no category filter)
        so MCP fallback and Atom API both return the broadest possible results.
        """
        raw = await ArXivMcpSource().search(
            query,
            max_results=max_results,
            # Do NOT default to [namespace_key] — that would apply a category
            # filter and exclude interdisciplinary results or fail entirely when
            # namespace_key is not a valid arXiv category string.
            namespace_keys=namespace_keys or None,
        )
        new_papers, skipped = await self.import_raw_papers(
            raw,
            namespace_key=namespace_key,
            create_embeddings=True,
            update_graph=True,
        )
        return new_papers, skipped, [self._raw_to_dict(p) for p in raw]

    async def _ensure_source_mapping(self, namespace_key: str) -> None:
        result = await self._db.execute(
            select(SourceMapping).where(
                SourceMapping.namespace_key == namespace_key,
                SourceMapping.source_name == "arxiv_mcp",
            )
        )
        if result.scalar_one_or_none():
            return
        arxiv_cat = NamespaceManager().arxiv_category(namespace_key) or NAMESPACE_TO_ARXIV.get(namespace_key) or namespace_key
        self._db.add(SourceMapping(
            namespace_key=namespace_key,
            source_name="arxiv_mcp",
            external_category_key=arxiv_cat,
        ))
        await self._db.flush()

    async def _embed_abstracts(self, papers: list[Paper]) -> None:
        try:
            from app.adapters.embedding import get_embedding_adapter
            from app.services.semantic_chunker import chunk_section

            embed = get_embedding_adapter()
            # Boundary-aware sub-chunking: most abstracts are short and emit
            # a single chunk, but long abstracts (4–5 paragraphs, sometimes
            # 3000+ chars from arXiv preprints) get split on paragraph /
            # sentence boundaries so retrieval doesn't blur multiple ideas
            # into one averaged embedding.
            flat_chunks: list[tuple[Paper, int, str]] = []
            for paper in papers:
                raw = paper.abstract or paper.title or ""
                for idx, sc in enumerate(chunk_section(raw, "abstract")):
                    flat_chunks.append((paper, idx, sc.content))
            if not flat_chunks:
                return
            texts = [c[2] for c in flat_chunks]
            vectors = await embed.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
            for (paper, idx, content), vec in zip(flat_chunks, vectors):
                if not vec:
                    continue
                self._db.add(PaperChunk(
                    paper_id=paper.id,
                    chunk_index=idx,
                    section_type="abstract",
                    content=content,
                    embedding=vec,
                    embedding_dim=embed.dimensions,
                    embedding_provider=embed.provider_id,
                ))
            await self._db.commit()
        except Exception as exc:
            log.warning("arxiv_import: embedding skipped: %s", exc)
            await self._db.rollback()

    async def _index_graph(self, papers: list[Paper]) -> None:
        # Hard-gated on the global graph_enabled flag — if the admin has
        # turned the graph off, we silently skip indexing. Imports still
        # succeed and embeddings still happen; only the knowledge-graph
        # write is suppressed.
        try:
            from app.services.admin_settings import get_app_settings
            settings_snapshot = await get_app_settings(self._db)
            if not bool(settings_snapshot.get("graph_enabled", False)):
                log.debug("arxiv_import: graph indexing skipped (feature disabled)")
                return
        except Exception:
            return  # fail-closed when we can't check the flag

        try:
            svc = GraphService(self._db)
            for paper in papers:
                await svc.add_paper_node(paper)
            await self._db.commit()
        except Exception as exc:
            log.warning("arxiv_import: graph indexing skipped: %s", exc)
            await self._db.rollback()

    @staticmethod
    def _raw_to_dict(paper: RawPaper) -> dict:
        return {
            "external_id": paper.external_id,
            "namespace_key": paper.namespace_key,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "source_url": paper.source_url,
            "pdf_url": paper.pdf_url,
            "published_at": paper.published_at.isoformat() if paper.published_at else None,
            "raw": paper.raw,
        }

    @staticmethod
    def _coerce_dt(value: object) -> datetime | None:
        if value is None or isinstance(value, datetime):
            if isinstance(value, datetime) and value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
