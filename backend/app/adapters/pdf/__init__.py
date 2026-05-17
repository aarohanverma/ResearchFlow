"""PDF adapter package — parser factory and fallback chain.

Default parser is **Marker** (memory-friendly, reliable on laptops/WSL).
Docling and Gemini Vision are available as opt-ins via ``PDF_PARSER``.

``parse_with_fallback`` always tries the chosen parser first, then falls
back through the rest of the installed chain so a failure at any tier is
transparent to callers.
"""

import asyncio
import logging

from app.adapters.pdf.base import FigureRef, ParsedPaper, PDFParser, Reference, Section
from app.adapters.pdf.gemini_vision import GeminiVisionFallback
from app.adapters.pdf.marker_parser import MarkerParser
from app.core.config import settings

log = logging.getLogger(__name__)

# Re-export schema types so callers only need to import from this package
__all__ = [
    "PDFParser",
    "ParsedPaper",
    "Section",
    "Reference",
    "FigureRef",
    "get_pdf_parser",
    "parse_with_fallback",
]


def get_pdf_parser() -> PDFParser:
    """Return the configured PDF parser backend.

    Selection logic:

    - ``PDF_PARSER=marker`` (default) → MarkerParser. Memory-safe on laptops.
    - ``PDF_PARSER=docling``           → DoclingParser if installed; otherwise
      logs a warning and falls back to MarkerParser. Docling is RAM-heavy
      and is opt-in to avoid OOM-crashes on small VMs.
    - ``PDF_PARSER=gemini_vision``     → GeminiVisionFallback.
    - Any unrecognised value           → MarkerParser.

    Returns:
        Configured ``PDFParser`` instance.
    """
    parser_name = (settings.pdf_parser or "marker").lower()

    if parser_name == "docling":
        try:
            from app.adapters.pdf.docling_parser import DoclingParser, _DOCLING_AVAILABLE
            if _DOCLING_AVAILABLE:
                return DoclingParser()
            log.warning(
                "PDF_PARSER=docling but docling is not installed — "
                "falling back to MarkerParser"
            )
        except ImportError:
            log.warning("DoclingParser import failed — falling back to MarkerParser")
        return MarkerParser()

    if parser_name == "gemini_vision":
        return GeminiVisionFallback()

    # marker (default) and any unknown value
    return MarkerParser()


async def parse_with_fallback(pdf_bytes: bytes) -> ParsedPaper:
    """Parse PDF bytes with a multi-tier fallback chain.

    The chosen parser (``PDF_PARSER``) leads. Remaining installed parsers
    follow in this fallback order:
    Marker → Docling (only if explicitly selected) → Gemini Vision.

    Each tier has a wall-clock timeout to prevent stalling the workflow:

    - Marker:    90 s (includes model download on first call)
    - Docling:   60 s
    - Gemini:    120 s

    If every parser fails, returns a minimal ``ParsedPaper`` so downstream
    code always has *something* to work with rather than raising.

    Args:
        pdf_bytes: Raw PDF bytes to parse.

    Returns:
        Best-effort ``ParsedPaper``.
    """
    chosen = (settings.pdf_parser or "marker").lower()
    chain: list[tuple[str, PDFParser, float]] = []
    seen: set[str] = set()

    def _push(name: str, factory):
        """Append a parser to the fallback chain if not already present.

        Silently skips the parser when instantiation fails (e.g. missing
        optional dependency). Each parser name is added at most once.

        Args:
            name: Canonical parser name (``"marker"``, ``"docling"``, etc.).
            factory: Two-tuple ``(ParserClass, timeout_seconds)`` — the
                class is instantiated here; the timeout is used by the caller.
        """
        if name in seen:
            return
        try:
            chain.append((name, factory[0](), factory[1]))
            seen.add(name)
        except Exception as exc:  # adapter failed to instantiate
            log.debug("parse_with_fallback skip %s: %s", name, exc)

    # 1. The user-selected parser, if available
    if chosen == "docling":
        try:
            from app.adapters.pdf.docling_parser import DoclingParser, _DOCLING_AVAILABLE
            if _DOCLING_AVAILABLE:
                _push("docling", (DoclingParser, 60.0))
        except ImportError:
            log.debug("parse_with_fallback: docling import failed")
    elif chosen == "gemini_vision":
        _push("gemini_vision", (GeminiVisionFallback, 120.0))
    else:  # marker or unknown
        _push("marker", (MarkerParser, 90.0))

    # 2. Marker is always part of the chain — most reliable second choice
    _push("marker", (MarkerParser, 90.0))

    # 3. Gemini Vision OCR — last-resort, requires GOOGLE_API_KEY
    _push("gemini_vision", (GeminiVisionFallback, 120.0))

    # 4. Docling only joins the chain if the user explicitly selected it.
    #    We deliberately do NOT auto-add it as a silent fallback, because
    #    its first-run model download and EasyOCR deps can spike RAM and
    #    crash low-memory hosts (e.g. WSL / small Docker VMs).

    last_exc: Exception | None = None
    for parser_name, parser, timeout in chain:
        try:
            result = await asyncio.wait_for(parser.parse(pdf_bytes), timeout=timeout)
            result.parser_name = parser_name
            result.fallback_used = parser_name != chain[0][0]
            log.info("parse_with_fallback succeeded parser=%s", parser_name)
            return result
        except asyncio.TimeoutError:
            log.warning("parse_with_fallback timeout parser=%s", parser_name)
            last_exc = TimeoutError(f"{parser_name} timed out after {timeout}s")
        except Exception as exc:
            log.warning("parse_with_fallback failed parser=%s err=%s", parser_name, exc)
            last_exc = exc

    log.error("parse_with_fallback all parsers failed last_err=%s", last_exc)
    return ParsedPaper(
        title="(parse failed)",
        sections=[Section(
            section_type="abstract",
            content="PDF could not be parsed. Using abstract from metadata.",
        )],
        references=[],
        figures=[],
        parser_name="none",
        fallback_used=True,
        parser_confidence=0.0,
    )
