"""Regression tests for two production-hardening fixes.

1. ``parse_with_fallback`` must reject non-PDF input (HTML error pages,
   redirect stubs, empty bodies) *before* invoking any parser, so a bad
   network fetch can never poison the RAG index with hallucinated chunks.

2. The podcast ``_AUDIO_BUFFER`` side-channel must stay bounded — a run that
   fails or is cancelled between the synthesize and save nodes would otherwise
   strand multi-MB audio in the process forever.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.adapters.pdf import parse_with_fallback
from app.adapters.pdf.base import ParsedPaper, Section


# ── 1. parse_with_fallback magic-byte guard ──────────────────────────────────


@pytest.mark.asyncio
async def test_parse_with_fallback_rejects_html_without_invoking_parser():
    """HTML bytes (a 404/landing page) short-circuit to the minimal result.

    Crucially, no real parser is constructed or called — the guard runs first,
    so we patch both parsers with AsyncMocks and assert they were never used.
    """
    html = b"<!DOCTYPE html><html><body>404 Not Found</body></html>"

    with patch("app.adapters.pdf.MarkerParser") as marker_cls, \
         patch("app.adapters.pdf.GeminiVisionFallback") as gemini_cls:
        result = await parse_with_fallback(html)
        marker_cls.assert_not_called()
        gemini_cls.assert_not_called()

    assert isinstance(result, ParsedPaper)
    assert result.parser_name == "none"
    assert result.parser_confidence == 0.0
    # The single section is an abstract sentinel — content_loader skips
    # abstract-typed sections, so nothing garbage is persisted.
    assert all(s.section_type == "abstract" for s in result.sections)


@pytest.mark.asyncio
async def test_parse_with_fallback_rejects_empty_bytes():
    """Empty bodies (redirect stubs with no content) are rejected too."""
    with patch("app.adapters.pdf.MarkerParser") as marker_cls, \
         patch("app.adapters.pdf.GeminiVisionFallback") as gemini_cls:
        result = await parse_with_fallback(b"")
        marker_cls.assert_not_called()
        gemini_cls.assert_not_called()
    assert result.parser_name == "none"


@pytest.mark.asyncio
async def test_parse_with_fallback_accepts_pdf_magic():
    """Bytes carrying the ``%PDF`` signature pass the guard into the parser."""
    pdf_like = b"%PDF-1.7\n" + b"0" * 200

    fake = ParsedPaper(
        title="Real Paper",
        sections=[Section(section_type="method", content="we did things")],
        references=[], figures=[],
        parser_name="marker", fallback_used=False, parser_confidence=0.9,
    )

    # Patch the marker parser instance's async parse to avoid heavy deps.
    with patch("app.adapters.pdf.settings") as fake_settings, \
         patch("app.adapters.pdf.MarkerParser") as marker_cls:
        fake_settings.pdf_parser = "marker"
        marker_cls.return_value.parse = AsyncMock(return_value=fake)
        result = await parse_with_fallback(pdf_like)

    assert result.parser_name == "marker"
    assert result.sections[0].section_type == "method"


@pytest.mark.asyncio
async def test_parse_with_fallback_accepts_pdf_with_leading_bytes():
    """A few junk bytes before ``%PDF`` (within 1 KB) still count as a PDF."""
    pdf_like = b"\x00\x00junk" + b"%PDF-1.4\n" + b"x" * 50
    fake = ParsedPaper(
        title="t", sections=[Section(section_type="results", content="r")],
        references=[], figures=[],
        parser_name="marker", fallback_used=False, parser_confidence=0.5,
    )
    with patch("app.adapters.pdf.settings") as fake_settings, \
         patch("app.adapters.pdf.MarkerParser") as marker_cls:
        fake_settings.pdf_parser = "marker"
        marker_cls.return_value.parse = AsyncMock(return_value=fake)
        result = await parse_with_fallback(pdf_like)
    assert result.parser_name == "marker"


# ── 2. Bounded podcast audio buffer ──────────────────────────────────────────


def _reset_audio_buffer():
    from app.workflows import podcast
    podcast._AUDIO_BUFFER.clear()


def test_audio_buffer_put_pop_roundtrip():
    """A put/pop pair returns the exact bytes and leaves the buffer empty."""
    from app.workflows import podcast
    _reset_audio_buffer()
    podcast._audio_buffer_put("art-1", b"hello-audio")
    assert podcast._audio_buffer_pop("art-1") == b"hello-audio"
    assert podcast._audio_buffer_pop("art-1") is None  # already removed
    assert len(podcast._AUDIO_BUFFER) == 0


def test_audio_buffer_enforces_hard_cap():
    """Writing past the cap evicts oldest entries so the map stays bounded.

    Simulates abandoned runs (entries written but never popped) and asserts
    the buffer never exceeds ``_AUDIO_BUFFER_MAX``.
    """
    from app.workflows import podcast
    _reset_audio_buffer()
    n = podcast._AUDIO_BUFFER_MAX + 25
    for i in range(n):
        podcast._audio_buffer_put(f"art-{i}", bytes([i % 256]) * 16)
    assert len(podcast._AUDIO_BUFFER) <= podcast._AUDIO_BUFFER_MAX
    # The most recent write must survive eviction.
    assert podcast._audio_buffer_pop(f"art-{n - 1}") is not None


def test_audio_buffer_evicts_stale_entries(monkeypatch):
    """Entries older than the TTL are reclaimed on the next write."""
    from app.workflows import podcast
    _reset_audio_buffer()

    clock = {"t": 1000.0}
    monkeypatch.setattr(podcast.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(podcast, "_AUDIO_BUFFER_TTL_S", 60.0)

    podcast._audio_buffer_put("stale", b"old")
    clock["t"] += 120.0  # advance well past the TTL
    podcast._audio_buffer_put("fresh", b"new")

    assert "stale" not in podcast._AUDIO_BUFFER
    assert podcast._audio_buffer_pop("fresh") == b"new"
