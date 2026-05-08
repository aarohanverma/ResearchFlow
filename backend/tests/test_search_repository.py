"""Unit tests for SearchRepository — keyword search, semantic search, and RRF fusion."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories.search import SearchRepository, _RRF_K, _KW_WEIGHT, _SEM_WEIGHT


def _make_row(paper_id: str, title: str, score: float = 0.8) -> dict:
    """Return a plain dict matching what SearchRepository._keyword_search returns."""
    return {
        "paper_id": paper_id,
        "external_id": f"ext_{paper_id}",
        "title": title,
        "abstract": f"Abstract for {title}",
        "authors": ["Author One"],
        "namespace_key": "cs.AI",
        "source_url": "https://arxiv.org/abs/test",
        "pdf_url": None,
        "novelty_score": score,
        "relevance_score": score,
        "is_breakthrough": False,
        "key_concepts": ["AI"],
        "methods_used": ["transformer"],
        "implications": None,
        "published_at": None,
        "ingested_at": None,
        "tldr": None,
        "kw_score": score,
        "sem_score": score,
    }


def _make_db_row(paper_id: str, title: str, score: float = 0.8) -> MagicMock:
    """Return a MagicMock with ._mapping for testing raw DB result rows."""
    row = MagicMock()
    row._mapping = _make_row(paper_id, title, score)
    return row


class TestRRFFusion:
    def _repo(self):
        return SearchRepository(AsyncMock())

    def test_keyword_only_results_ranked_correctly(self):
        repo = self._repo()
        kw = [_make_row("a", "Paper A"), _make_row("b", "Paper B")]
        fused = repo._rrf_fuse(kw, [])

        assert len(fused) == 2
        # First result should have higher score (lower rank = higher RRF score)
        assert fused[0]["search_score"] > fused[1]["search_score"]
        assert fused[0]["paper_id"] == "a"

    def test_semantic_only_results_excluded(self):
        # Semantic-only results are excluded by design: semantic re-ranks keyword
        # results but never adds papers that don't appear in keyword results.
        repo = self._repo()
        sem = [_make_row("x", "Paper X"), _make_row("y", "Paper Y")]
        fused = repo._rrf_fuse([], sem)

        assert len(fused) == 0

    def test_hybrid_result_gets_both_rank_scores(self):
        repo = self._repo()
        # Paper "a" appears in both keyword and semantic
        kw = [_make_row("a", "Shared Paper"), _make_row("b", "KW-only Paper")]
        sem = [_make_row("a", "Shared Paper"), _make_row("c", "SEM-only Paper")]
        fused = repo._rrf_fuse(kw, sem)

        shared = next(r for r in fused if r["paper_id"] == "a")
        kw_only = next(r for r in fused if r["paper_id"] == "b")

        # Shared paper has two RRF contributions → higher score
        assert shared["search_score"] > kw_only["search_score"]
        assert shared["match_type"] == "hybrid"

    def test_rrf_score_formula(self):
        repo = self._repo()
        kw = [_make_row("a", "Paper A")]  # rank 1
        sem = [_make_row("a", "Paper A")]  # rank 1
        fused = repo._rrf_fuse(kw, sem)

        expected = round(_KW_WEIGHT / (_RRF_K + 1) + _SEM_WEIGHT / (_RRF_K + 1), 6)
        assert abs(fused[0]["search_score"] - expected) < 1e-5

    def test_empty_both_returns_empty(self):
        repo = self._repo()
        assert repo._rrf_fuse([], []) == []

    def test_match_type_keyword_when_only_in_kw(self):
        repo = self._repo()
        kw = [_make_row("a", "KW paper")]
        fused = repo._rrf_fuse(kw, [])
        assert fused[0]["match_type"] == "keyword"


class TestKeywordSearch:
    @pytest.mark.asyncio
    async def test_returns_results_from_db(self):
        db = AsyncMock()
        rows = [_make_db_row("p1", "Neural Networks"), _make_db_row("p2", "Attention Mechanism")]
        result = MagicMock()
        result.fetchall.return_value = rows
        db.execute = AsyncMock(return_value=result)

        repo = SearchRepository(db)
        results = await repo._keyword_search("neural attention")

        assert len(results) == 2
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_namespace_filter_passed_when_given(self):
        db = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = []
        db.execute = AsyncMock(return_value=result)

        repo = SearchRepository(db)
        await repo._keyword_search("transformers", namespace_keys=["cs.AI"])

        # The first execute call should include ns0 param with the namespace value
        first_call_params = db.execute.call_args_list[0][0][1]
        assert "ns0" in first_call_params
        assert first_call_params["ns0"] == "cs.AI"

    @pytest.mark.asyncio
    async def test_db_error_returns_empty_list(self):
        db = AsyncMock()
        db.execute = AsyncMock(side_effect=Exception("DB connection failed"))

        repo = SearchRepository(db)
        results = await repo._keyword_search("anything")
        assert results == []


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_keyword_only_when_no_vector(self):
        db = AsyncMock()
        rows = [_make_db_row("p1", "Test Paper")]
        result = MagicMock()
        result.fetchall.return_value = rows
        db.execute = AsyncMock(return_value=result)

        repo = SearchRepository(db)
        results = await repo.hybrid_search("deep learning", query_vector=None)

        assert len(results) == 1
        # execute called at least once (keyword search); no semantic path
        assert db.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_hybrid_calls_both_searches(self):
        db = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = []
        db.execute = AsyncMock(return_value=result)

        repo = SearchRepository(db)
        dummy_vector = [0.1] * 768
        await repo.hybrid_search("attention", query_vector=dummy_vector)

        # Both keyword and semantic paths must be exercised (≥2 DB calls)
        assert db.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_limit_applied_to_final_results(self):
        rows = [_make_db_row(f"p{i}", f"Paper {i}") for i in range(30)]
        db = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = rows
        db.execute = AsyncMock(return_value=result)

        repo = SearchRepository(db)
        results = await repo.hybrid_search("test", limit=5)
        assert len(results) <= 5
