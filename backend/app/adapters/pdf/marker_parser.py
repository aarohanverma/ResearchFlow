"""Marker PDF parser — best-in-class for two-column academic layouts."""

import asyncio
import io
import tempfile

from app.adapters.pdf.base import FigureRef, ParsedPaper, PDFParser, Reference, Section

_SECTION_KEYWORDS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "method": "methodology",
    "methodology": "methodology",
    "approach": "methodology",
    "experiment": "results",
    "result": "results",
    "evaluation": "results",
    "discussion": "discussion",
    "limitation": "limitations",
    "conclusion": "conclusion",
    "reference": "references",
}


def _classify_section(heading: str) -> str:
    """Map a section heading string to a canonical section-type label."""
    lower = heading.lower()
    for kw, label in _SECTION_KEYWORDS.items():
        if kw in lower:
            return label
    return "other"


class MarkerParser(PDFParser):
    """Uses the marker-pdf library for academic PDF parsing."""

    async def parse(self, pdf_bytes: bytes) -> ParsedPaper:
        """Parse a PDF asynchronously using the Marker library.

        Delegates to ``_parse_sync`` in a thread-pool executor to avoid
        blocking the async event loop.

        Args:
            pdf_bytes: Raw bytes of the PDF file to parse.

        Returns:
            A ``ParsedPaper`` containing title, sections, references, and
            figures extracted by Marker.
        """
        return await asyncio.get_event_loop().run_in_executor(None, self._parse_sync, pdf_bytes)

    def _parse_sync(self, pdf_bytes: bytes) -> ParsedPaper:
        """Run the synchronous Marker conversion pipeline on raw PDF bytes."""
        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
        except ImportError:
            raise RuntimeError("marker-pdf not installed — add it to requirements.txt")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        models = load_all_models()
        full_text, images, metadata = convert_single_pdf(tmp_path, models)

        sections = self._parse_sections(full_text)
        references = self._extract_references(sections)
        figures = self._extract_figures(images)

        title = metadata.get("title", sections[0].content[:120] if sections else "Unknown")
        return ParsedPaper(title=title, sections=sections, references=references, figures=figures)

    def _parse_sections(self, text: str) -> list[Section]:
        """Heuristic section splitter — splits on markdown headings from Marker."""
        import re
        chunks = re.split(r"\n#{1,3} ", text)
        sections: list[Section] = []
        for i, chunk in enumerate(chunks):
            lines = chunk.strip().split("\n", 1)
            heading = lines[0].strip() if lines else "body"
            body = lines[1].strip() if len(lines) > 1 else chunk.strip()

            section_type = _classify_section(heading) if i > 0 else "abstract"
            math_blocks = re.findall(r"\$\$(.+?)\$\$", body, re.DOTALL)
            sections.append(Section(section_type=section_type, content=body, math_blocks=math_blocks))

        return sections

    def _extract_references(self, sections: list[Section]) -> list[Reference]:
        """Rudimentary reference extraction from the references section."""
        import re
        refs: list[Reference] = []
        for s in sections:
            if s.section_type == "references":
                # Match arXiv IDs
                arxiv_ids = re.findall(r"arXiv:(\d{4}\.\d{4,5})", s.content)
                for aid in arxiv_ids:
                    refs.append(Reference(title="", arxiv_id=aid))
                break
        return refs

    def _extract_figures(self, images: dict) -> list[FigureRef]:
        """Convert the Marker image-dict into a list of ``FigureRef`` objects."""
        figures = []
        for key, img_data in (images or {}).items():
            caption = img_data.get("caption", "") if isinstance(img_data, dict) else ""
            figures.append(FigureRef(caption=caption, page_no=0))
        return figures
