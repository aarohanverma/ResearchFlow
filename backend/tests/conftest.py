"""Shared pytest fixtures for ResearchFlow backend tests."""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

# Point at test env before any app imports load settings
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")
os.environ.setdefault("ENVIRONMENT", "local")
os.environ.setdefault("DEBUG", "true")


@pytest.fixture
def mock_db():
    """Async mock SQLAlchemy session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.delete = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def paper_id():
    return uuid.uuid4()


@pytest.fixture
def user_id():
    return uuid.uuid4()


@pytest.fixture
def sample_paper(paper_id):
    """Mock Paper ORM object."""
    from app.models.paper import Paper  # noqa: deferred

    p = MagicMock(spec=Paper)
    p.id = paper_id
    p.external_id = "2401.00001"
    p.namespace_key = "cs.AI"
    p.title = "Attention Is All You Need: Revisited"
    p.authors = ["Alice Smith", "Bob Jones"]
    p.abstract = "We propose a novel transformer architecture for NLP tasks."
    p.source_url = "https://arxiv.org/abs/2401.00001"
    p.pdf_url = "https://arxiv.org/pdf/2401.00001.pdf"
    p.published_at = None
    p.key_concepts = ["transformer", "attention", "NLP"]
    p.methods_used = ["self-attention", "feed-forward"]
    p.implications = "Improves SOTA on GLUE by 3 points."
    p.novelty_score = 0.75
    p.relevance_score = 0.65
    p.is_breakthrough = False
    p.ingested_at = None
    return p


@pytest.fixture
def breakthrough_paper(paper_id):
    from app.models.paper import Paper

    p = MagicMock(spec=Paper)
    p.id = paper_id
    p.external_id = "2401.99999"
    p.namespace_key = "cs.AI"
    p.title = "Breakthrough: AGI Achieved"
    p.authors = ["C. Scientist"]
    p.abstract = "We present the first AGI."
    p.novelty_score = 0.92
    p.relevance_score = 0.88
    p.is_breakthrough = True
    p.namespace_key = "cs.AI"
    return p


@pytest.fixture
def mock_scalars_result(sample_paper):
    """Scalar result that returns one paper."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = [sample_paper]
    result.scalars.return_value.__iter__ = lambda self: iter([sample_paper])
    return result
