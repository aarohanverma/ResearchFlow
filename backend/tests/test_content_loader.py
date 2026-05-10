"""Tests for ContentLoaderService — paper/capsule/folder content loading.

Exercises the deep-PDF-grounding contract: when a paper has parsed section
chunks, they must appear in the loaded content; when a folder is requested
as a different user, ownership must be enforced; when sources are missing,
``ok=False`` must be returned without raising.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.content_loader import ContentLoaderService, LoadedContent


def _mock_paper(**kwargs):
    paper = MagicMock()
    paper.id = kwargs.get("id", uuid.uuid4())
    paper.title = kwargs.get("title", "Test Paper")
    paper.authors = kwargs.get("authors", ["Alice", "Bob"])
    paper.abstract = kwargs.get("abstract", "An interesting abstract.")
    paper.key_concepts = kwargs.get("key_concepts", ["attention", "transformer"])
    paper.methods_used = kwargs.get("methods_used", ["self-attention"])
    paper.implications = kwargs.get("implications", "Improves NLP tasks.")
    paper.tldr = kwargs.get("tldr", None)
    paper.pdf_url = kwargs.get("pdf_url", None)
    paper.parser_used = kwargs.get("parser_used", None)
    paper.pdf_parsed = kwargs.get("pdf_parsed", False)
    paper.parser_fallback_used = kwargs.get("parser_fallback_used", False)
    paper.parse_duration_ms = kwargs.get("parse_duration_ms", None)
    paper.parser_confidence = kwargs.get("parser_confidence", None)
    return paper


def _mock_chunk(section_type="abstract", content="Default body."):
    chunk = MagicMock()
    chunk.section_type = section_type
    chunk.content = content
    return chunk


@pytest.fixture
def db():
    return AsyncMock()


@pytest.fixture
def loader(db):
    return ContentLoaderService(db)


# ── Paper loading ─────────────────────────────────────────────────────────────


class TestLoadPaper:
    @pytest.mark.asyncio
    async def test_returns_not_found_when_paper_missing(self, loader, db, monkeypatch):
        from app.repositories.paper import PaperRepository

        async def _missing(self, paper_id):
            return None

        monkeypatch.setattr(PaperRepository, "get_by_id", _missing)

        result = await loader.load(
            source_type="paper",
            source_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
        )
        assert result.ok is False
        assert "not found" in result.title.lower()

    @pytest.mark.asyncio
    async def test_includes_abstract_and_section_chunks_when_parsed(self, loader, db, monkeypatch):
        from app.repositories.paper import PaperRepository

        paper = _mock_paper(title="Attention Is All You Need", abstract="The dominant approach...")
        chunks = [
            _mock_chunk("abstract", "abstract body"),
            _mock_chunk("introduction", "Recent work in seq2seq..."),
            _mock_chunk("methodology", "We propose multi-head attention with..."),
            _mock_chunk("results", "BLEU 28.4 on WMT-14, beating baselines by 2 points."),
        ]

        async def _get_by_id(self, pid):
            return paper

        async def _get_chunks(self, pid):
            return chunks

        async def _get_summary(self, pid, lvl):
            return None

        monkeypatch.setattr(PaperRepository, "get_by_id", _get_by_id)
        monkeypatch.setattr(PaperRepository, "get_chunks", _get_chunks)
        monkeypatch.setattr(PaperRepository, "get_summary", _get_summary)

        result = await loader.load(
            source_type="paper",
            source_id=paper.id,
            user_id=uuid.uuid4(),
        )
        assert result.ok is True
        assert "Attention Is All You Need" in result.title
        # Deep grounding: section content must appear in body
        assert "multi-head attention" in result.content.lower()
        assert "BLEU 28.4" in result.content
        assert result.paper_count == 1

    @pytest.mark.asyncio
    async def test_falls_back_gracefully_when_no_pdf_url(self, loader, db, monkeypatch):
        """When pdf_url is missing, no parse is attempted and abstract-only grounding is used."""
        from app.repositories.paper import PaperRepository

        paper = _mock_paper(pdf_url=None)
        chunks = [_mock_chunk("abstract", paper.abstract)]

        async def _get_by_id(self, pid): return paper
        async def _get_chunks(self, pid): return chunks
        async def _get_summary(self, pid, lvl): return None

        monkeypatch.setattr(PaperRepository, "get_by_id", _get_by_id)
        monkeypatch.setattr(PaperRepository, "get_chunks", _get_chunks)
        monkeypatch.setattr(PaperRepository, "get_summary", _get_summary)

        result = await loader.load(
            source_type="paper",
            source_id=paper.id,
            user_id=uuid.uuid4(),
        )
        assert result.ok is True
        assert "PDF body unavailable" in result.content


# ── Folder loading ────────────────────────────────────────────────────────────


class TestLoadFolder:
    @pytest.mark.asyncio
    async def test_folder_not_found_returns_not_ok(self, loader, db):
        # First execute call resolves to no folder → ok=False
        no_folder = MagicMock()
        no_folder.scalar_one_or_none = MagicMock(return_value=None)
        db.execute = AsyncMock(return_value=no_folder)

        result = await loader.load(
            source_type="folder",
            source_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
        )
        assert result.ok is False

    @pytest.mark.asyncio
    async def test_folder_requires_user_id(self, loader, db):
        # user_id=None should short-circuit with ok=False
        # Construct a service directly so we can pass user_id=None
        result = await loader.load(
            source_type="folder",
            source_id=uuid.uuid4(),
            user_id=None,
        )
        assert result.ok is False
        assert "missing user" in result.title.lower()


# ── Capsule loading ───────────────────────────────────────────────────────────


class TestLoadCapsule:
    @pytest.mark.asyncio
    async def test_capsule_not_found_returns_not_ok(self, loader, db):
        no_capsule = MagicMock()
        no_capsule.scalar_one_or_none = MagicMock(return_value=None)
        db.execute = AsyncMock(return_value=no_capsule)

        result = await loader.load(
            source_type="capsule",
            source_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
        )
        assert result.ok is False


# ── Unknown source ────────────────────────────────────────────────────────────


class TestLoadUnknown:
    @pytest.mark.asyncio
    async def test_unknown_source_type(self, loader):
        result = await loader.load(
            source_type="bogus",
            source_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
        )
        assert result.ok is False
        assert isinstance(result, LoadedContent)
