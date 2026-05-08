"""GeminiVisionFallback — page-image OCR + structure extraction for scanned PDFs."""

import asyncio
import base64

import google.generativeai as genai

from app.adapters.pdf.base import FigureRef, ParsedPaper, PDFParser, Reference, Section
from app.core.config import settings


class GeminiVisionFallback(PDFParser):
    """Converts each PDF page to an image and uses Gemini multimodal to extract text."""

    def __init__(self) -> None:
        """Configure the Gemini SDK and load the multimodal vision model."""
        genai.configure(api_key=settings.google_api_key)
        self._model = genai.GenerativeModel("gemini-3.1-pro")

    async def parse(self, pdf_bytes: bytes) -> ParsedPaper:
        """Parse a PDF asynchronously via Gemini multimodal OCR.

        Delegates to ``_parse_sync`` in a thread-pool executor to avoid
        blocking the async event loop.

        Args:
            pdf_bytes: Raw bytes of the PDF file to parse.

        Returns:
            A ``ParsedPaper`` with title and sections extracted from
            per-page OCR; references and figures are empty lists.
        """
        return await asyncio.get_event_loop().run_in_executor(None, self._parse_sync, pdf_bytes)

    def _parse_sync(self, pdf_bytes: bytes) -> ParsedPaper:
        """Render each PDF page to an image and extract text via Gemini Vision OCR."""
        try:
            import fitz  # PyMuPDF — for page rendering only (NOT content extraction)
        except ImportError:
            raise RuntimeError("PyMuPDF needed for page rendering in GeminiVisionFallback")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text_parts: list[str] = []

        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode()

            prompt = (
                "Extract the full text from this academic paper page. "
                "Preserve section headings using markdown (## Heading). "
                "Return only the extracted text — no commentary."
            )
            resp = self._model.generate_content(
                [{"mime_type": "image/png", "data": b64}, prompt]
            )
            full_text_parts.append(resp.text)

        combined = "\n\n".join(full_text_parts)

        # Re-use the section parsing from Marker
        from app.adapters.pdf.marker_parser import MarkerParser
        mp = MarkerParser()
        sections = mp._parse_sections(combined)
        title = sections[0].content[:120] if sections else "Unknown"
        return ParsedPaper(title=title, sections=sections, references=[], figures=[])
