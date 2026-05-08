import asyncio

from app.adapters.pdf.base import FigureRef, ParsedPaper, PDFParser, Reference, Section
from app.adapters.pdf.gemini_vision import GeminiVisionFallback
from app.adapters.pdf.marker_parser import MarkerParser
from app.core.config import settings


def get_pdf_parser() -> PDFParser:
    """Return the configured PDF parser backend.

    Returns:
        A ``MarkerParser`` if ``settings.pdf_parser`` is ``"marker"``,
        otherwise a ``GeminiVisionFallback`` instance.
    """
    if settings.pdf_parser == "marker":
        return MarkerParser()
    return GeminiVisionFallback()


async def parse_with_fallback(pdf_bytes: bytes) -> ParsedPaper:
    """Try Marker (90s timeout for model download), fall back to Gemini Vision."""
    try:
        return await asyncio.wait_for(MarkerParser().parse(pdf_bytes), timeout=90.0)
    except Exception:
        return await GeminiVisionFallback().parse(pdf_bytes)


__all__ = ["PDFParser", "ParsedPaper", "Section", "Reference", "FigureRef",
           "get_pdf_parser", "parse_with_fallback"]
