"""Unit tests for ScoringService — validates score formula and tag selection."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.user import Orientation
from app.services.scoring import ScoringService, _ORIENTATION_WEIGHT


class TestOrientationWeights:
    def test_research_weights_novelty_higher(self):
        assert _ORIENTATION_WEIGHT[Orientation.research] == 0.7

    def test_production_weights_relevance_higher(self):
        assert _ORIENTATION_WEIGHT[Orientation.production] == 0.3

    def test_both_balanced(self):
        assert _ORIENTATION_WEIGHT[Orientation.both] == 0.5


def _make_paper(novelty: float, relevance: float, ns: str = "cs.AI", breakthrough: bool = False):
    p = MagicMock()
    p.id = uuid.uuid4()
    p.namespace_key = ns
    p.novelty_score = novelty
    p.relevance_score = relevance
    p.is_breakthrough = breakthrough
    return p


def _make_service_with_papers(papers: list):
    """Returns a ScoringService whose DB execute returns the given papers."""
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value = papers
    db.execute = AsyncMock(return_value=result)
    return ScoringService(db)


class TestScoreFormula:
    @pytest.mark.asyncio
    async def test_basic_score_both_orientation(self):
        paper = _make_paper(novelty=0.8, relevance=0.6)
        svc = _make_service_with_papers([paper])
        scored = await svc.score_papers_for_user(
            user_id=uuid.uuid4(),
            namespace_key="cs.AI",
            orientation=Orientation.both,
        )
        assert len(scored) == 1
        expected = 0.8 * 0.5 + 0.6 * 0.5  # = 0.7
        assert abs(scored[0]["score"] - expected) < 0.001

    @pytest.mark.asyncio
    async def test_research_orientation_boosts_novelty(self):
        paper = _make_paper(novelty=1.0, relevance=0.0)
        svc = _make_service_with_papers([paper])
        scored = await svc.score_papers_for_user(
            user_id=uuid.uuid4(),
            namespace_key="cs.AI",
            orientation=Orientation.research,
        )
        # 1.0 * 0.7 + 0.0 * 0.3 = 0.7
        assert abs(scored[0]["score"] - 0.7) < 0.001

    @pytest.mark.asyncio
    async def test_production_orientation_boosts_relevance(self):
        paper = _make_paper(novelty=0.0, relevance=1.0)
        svc = _make_service_with_papers([paper])
        scored = await svc.score_papers_for_user(
            user_id=uuid.uuid4(),
            namespace_key="cs.AI",
            orientation=Orientation.production,
        )
        # 0.0 * 0.3 + 1.0 * 0.7 = 0.7
        assert abs(scored[0]["score"] - 0.7) < 0.001

    @pytest.mark.asyncio
    async def test_score_clamped_to_one(self):
        paper = _make_paper(novelty=1.0, relevance=1.0, ns="cs.AI")
        svc = _make_service_with_papers([paper])
        scored = await svc.score_papers_for_user(
            user_id=uuid.uuid4(),
            namespace_key="cs.AI",
            orientation=Orientation.both,
            hot_subtopics=["cs.AI"],  # +0.2 boost
        )
        assert scored[0]["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_zero(self):
        paper = _make_paper(novelty=0.05, relevance=0.05, ns="cs.AI")
        svc = _make_service_with_papers([paper])
        scored = await svc.score_papers_for_user(
            user_id=uuid.uuid4(),
            namespace_key="cs.AI",
            orientation=Orientation.both,
            cold_subtopics=["cs.AI"],  # -0.15 penalty
        )
        assert scored[0]["score"] >= 0.0

    @pytest.mark.asyncio
    async def test_results_sorted_descending(self):
        papers = [
            _make_paper(novelty=0.2, relevance=0.2),
            _make_paper(novelty=0.9, relevance=0.9),
            _make_paper(novelty=0.5, relevance=0.5),
        ]
        svc = _make_service_with_papers(papers)
        scored = await svc.score_papers_for_user(
            user_id=uuid.uuid4(),
            namespace_key="cs.AI",
            orientation=Orientation.both,
        )
        scores = [s["score"] for s in scored]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_namespace_returns_empty(self):
        svc = _make_service_with_papers([])
        scored = await svc.score_papers_for_user(
            user_id=uuid.uuid4(),
            namespace_key="cs.AI",
            orientation=Orientation.both,
        )
        assert scored == []


class TestWhyTag:
    def _svc(self):
        return ScoringService(AsyncMock())

    def test_high_novelty_tag(self):
        paper = _make_paper(novelty=0.85, relevance=0.5)
        tag = self._svc()._why_tag(paper, [])
        assert "novelty" in tag.lower() or "🔬" in tag

    def test_high_relevance_tag(self):
        paper = _make_paper(novelty=0.5, relevance=0.85)
        tag = self._svc()._why_tag(paper, [])
        assert "relevance" in tag.lower() or "🔧" in tag

    def test_breakthrough_tag(self):
        paper = _make_paper(novelty=0.5, relevance=0.5, breakthrough=True)
        tag = self._svc()._why_tag(paper, [])
        assert "breakthrough" in tag.lower() or "⚡" in tag

    def test_default_tag(self):
        paper = _make_paper(novelty=0.5, relevance=0.5)
        tag = self._svc()._why_tag(paper, [])
        assert tag  # non-empty
