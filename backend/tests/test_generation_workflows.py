"""Tests for media generation workflows (podcast, slides).

All external calls (LLM, TTS, DB, blob storage) are mocked so tests run
without API keys or a real database.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.adapters.llm.base import CompletionResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_completion(text: str) -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=100,
        output_tokens=200,
        model_used="gpt-4o-mini",
        provider_used="openai",
        latency_ms=500,
    )


def _json_result(obj: dict | list) -> CompletionResult:
    return _make_completion(json.dumps(obj))


# ── PDF Parser Base Tests ──────────────────────────────────────────────────────

class TestParsedPaperBackwardCompat:
    """Verify that the enhanced ParsedPaper schema is backward-compatible."""

    def test_minimal_construction_works(self):
        """Old code that only sets title/sections/references/figures must still work."""
        from app.adapters.pdf.base import ParsedPaper, Section

        paper = ParsedPaper(
            title="My Paper",
            sections=[Section(section_type="abstract", content="Hello world")],
            references=[],
            figures=[],
        )

        assert paper.title == "My Paper"
        assert len(paper.sections) == 1
        assert paper.tables == []
        assert paper.equations == []
        assert paper.abstract is None
        assert paper.parser_name == "unknown"
        assert paper.fallback_used is False

    def test_section_backward_compat(self):
        """Section with only section_type + content still works."""
        from app.adapters.pdf.base import Section

        s = Section(section_type="introduction", content="Text here")

        assert s.section_type == "introduction"
        assert s.content == "Text here"
        assert s.math_blocks == []
        assert s.tables == []
        assert s.heading is None
        assert s.level == 1
        assert s.figures == []

    def test_chunk_text_utility(self):
        """TTSAdapter.chunk_text splits long text at sentence boundaries."""
        from app.adapters.tts.base import TTSAdapter

        class _DummyTTS(TTSAdapter):
            provider_id = "test"
            default_voice = "v1"

            @property
            def max_chars_per_call(self) -> int:
                return 50

            async def synthesize(self, text, **kwargs):
                pass

        tts = _DummyTTS()
        text = "First sentence. Second sentence. Third sentence. Fourth one here."
        chunks = tts.chunk_text(text, max_chars=30)

        assert all(len(c) <= 30 for c in chunks)
        assert "".join(chunks).replace(". ", ".").replace("  ", " ")


# ── Marp Slide Count ───────────────────────────────────────────────────────────

class TestSlidesAdapter:
    def test_count_slides_single(self):
        from app.adapters.slides.base import SlidesAdapter

        md = "# Slide 1\nContent here"
        assert SlidesAdapter.count_slides(md) == 1

    def test_count_slides_multiple(self):
        from app.adapters.slides.base import SlidesAdapter

        md = "# Slide 1\nContent\n---\n# Slide 2\nMore\n---\n# Slide 3"
        assert SlidesAdapter.count_slides(md) == 3

    def test_count_slides_empty(self):
        from app.adapters.slides.base import SlidesAdapter

        assert SlidesAdapter.count_slides("") == 1


# ── Podcast Utterance Parser ───────────────────────────────────────────────────

class TestPodcastScriptParser:
    def test_parse_utterances_basic(self):
        from app.workflows.podcast import _parse_utterances

        script = "[HOST]: Hello, welcome to the show.\n[EXPERT]: Great to be here.\n[HOST]: Let's dive in."
        utts = _parse_utterances(script, "practitioner")

        assert len(utts) == 3
        assert utts[0]["speaker"] == "HOST"
        assert utts[1]["speaker"] == "EXPERT"
        assert "Hello" in utts[0]["text"]

    def test_parse_utterances_ignores_blank_lines(self):
        from app.workflows.podcast import _parse_utterances

        script = "[HOST]: Line one.\n\n[EXPERT]: Line two.\n\n"
        utts = _parse_utterances(script, "newcomer")

        assert len(utts) == 2

    def test_voice_assignment_expertise(self):
        from app.adapters.tts import EXPERTISE_VOICES
        from app.workflows.podcast import _parse_utterances

        script = "[HOST]: Welcome.\n[EXPERT]: Thanks."
        utts_newcomer = _parse_utterances(script, "newcomer")
        utts_expert = _parse_utterances(script, "expert")

        assert utts_newcomer[0]["voice"] == EXPERTISE_VOICES["newcomer"]
        assert utts_expert[0]["voice"] == EXPERTISE_VOICES["expert"]
        # EXPERT voice is always the same
        assert utts_newcomer[1]["voice"] == utts_expert[1]["voice"]



class TestPodcastWorkflow:
    @pytest.mark.asyncio
    async def test_plan_episode_uses_llm(self):
        """_plan_episode should call LLM and parse JSON episode plan."""
        from app.workflows.podcast import _plan_episode, PodcastState

        plan_json = {
            "episode_title": "Test Episode",
            "tagline": "Fascinating research",
            "host_name": "Alex",
            "expert_name": "Dr. Smith",
            "estimated_minutes": 15,
            "segments": [{"segment_id": 1, "title": "Intro", "talking_points": [], "duration_minutes": 2}],
        }

        mock_llm = MagicMock()
        mock_llm.quality_model = "gpt-4o-mini"
        mock_llm.complete = AsyncMock(return_value=_json_result(plan_json))

        with patch("app.workflows.podcast.get_llm_adapter", return_value=mock_llm):
            state: PodcastState = {
                "artifact_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "source_type": "paper",
                "source_id": str(uuid.uuid4()),
                "expertise_level": "practitioner",
                "orientation": "both",
                "title": "Test Paper",
                "paper_content": "Abstract: This is a test paper.",
                "episode_plan": {},
                "script": "",
                "utterances": [],
                "audio_bytes": None,
                "blob_path": None,
                "error_metadata": {},
            }

            result = await _plan_episode(state)

        assert result["episode_plan"]["episode_title"] == "Test Episode"
        assert len(result["episode_plan"]["segments"]) == 1
        mock_llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_synthesize_audio_skips_on_missing_key(self):
        """_synthesize_audio should gracefully skip when OPENAI_API_KEY is absent."""
        from app.workflows.podcast import _synthesize_audio, PodcastState

        with patch("app.core.config.settings") as mock_settings:
            mock_settings.openai_api_key = ""
            state: PodcastState = {
                "artifact_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "source_type": "paper",
                "source_id": str(uuid.uuid4()),
                "expertise_level": "practitioner",
                "orientation": "both",
                "title": "Test",
                "paper_content": "content",
                "episode_plan": {},
                "script": "[HOST]: Hello.\n[EXPERT]: Hi.",
                "utterances": [],
                "audio_bytes": None,
                "blob_path": None,
                "error_metadata": {},
            }

            result = await _synthesize_audio(state)

        assert result["audio_bytes"] is None
        assert result["utterances"] == []


# ── Slides workflow unit tests ─────────────────────────────────────────────────

class TestSlidesWorkflow:
    @pytest.mark.asyncio
    async def test_plan_slides_returns_plan(self):
        """_plan_slides should call LLM and parse JSON slide plan."""
        from app.workflows.slides import _plan_slides, SlidesState

        slide_plan = {
            "deck_title": "Deep Learning for NLP",
            "subtitle": "A survey of recent advances",
            "theme": "gaia",
            "total_slides": 14,
            "slides": [{"slide_num": 1, "type": "title"}],
        }

        mock_llm = MagicMock()
        mock_llm.quality_model = "gpt-4o-mini"
        mock_llm.complete = AsyncMock(return_value=_json_result(slide_plan))

        with patch("app.workflows.slides.get_llm_adapter", return_value=mock_llm):
            state: SlidesState = {
                "artifact_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "source_type": "paper",
                "source_id": str(uuid.uuid4()),
                "expertise_level": "expert",
                "orientation": "research",
                "title": "NLP Paper",
                "paper_content": "Abstract text here.",
                "slide_plan": {},
                "marp_markdown": "",
                "blob_path": None,
                "error_metadata": {},
            }

            result = await _plan_slides(state)

        assert result["slide_plan"]["deck_title"] == "Deep Learning for NLP"
        assert result["slide_plan"]["total_slides"] == 14

    @pytest.mark.asyncio
    async def test_write_markdown_produces_marp(self):
        """_write_markdown should call reasoning_model and store markdown."""
        from app.workflows.slides import _write_markdown, SlidesState

        sample_md = "---\nmarp: true\n---\n\n# Title\n\n---\n\n# Slide 2"

        mock_llm = MagicMock()
        mock_llm.reasoning_model = "gpt-4o"
        mock_llm.complete = AsyncMock(return_value=_make_completion(sample_md))

        with patch("app.workflows.slides.get_llm_adapter", return_value=mock_llm):
            state: SlidesState = {
                "artifact_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "source_type": "paper",
                "source_id": str(uuid.uuid4()),
                "expertise_level": "practitioner",
                "orientation": "both",
                "title": "Paper",
                "paper_content": "content",
                "slide_plan": {"theme": "gaia", "total_slides": 2, "slides": []},
                "marp_markdown": "",
                "blob_path": None,
                "error_metadata": {},
            }

            result = await _write_markdown(state)

        assert "marp: true" in result["marp_markdown"]


