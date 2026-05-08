"""PDFParser ABC — Marker is primary, GeminiVision is fallback."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Reference:
    """A bibliographic reference extracted from a paper.

    Attributes:
        title: Reference title string (may be empty if only an arXiv ID was found).
        authors: List of author name strings.
        arxiv_id: arXiv paper identifier if present (e.g. ``"2301.07041"``).
    """

    title: str
    authors: list[str] = field(default_factory=list)
    arxiv_id: str | None = None


@dataclass
class FigureRef:
    """Metadata for a figure extracted from a PDF page.

    Attributes:
        caption: Figure caption text.
        page_no: Zero-based page index where the figure appears.
        bbox: Bounding box coordinates ``[x0, y0, x1, y1]``.
        image_bytes: Raw PNG/JPEG bytes of the figure, if extracted.
    """

    caption: str
    page_no: int
    bbox: list[float] = field(default_factory=list)   # [x0, y0, x1, y1]
    image_bytes: bytes | None = None


@dataclass
class Section:
    """A logical section of a parsed academic paper.

    Attributes:
        section_type: Canonical section label (e.g. ``"abstract"``,
            ``"introduction"``, ``"methodology"``, ``"results"``).
        content: Full text of the section body.
        math_blocks: List of LaTeX math expressions found in the section.
        tables: List of raw table strings extracted from the section.
    """

    section_type: str     # abstract | introduction | methodology | results | ...
    content: str
    math_blocks: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


@dataclass
class ParsedPaper:
    """Structured representation of a fully parsed academic paper.

    Attributes:
        title: Paper title string.
        sections: Ordered list of extracted sections.
        references: List of bibliographic references.
        figures: List of figure metadata objects.
    """

    title: str
    sections: list[Section]
    references: list[Reference]
    figures: list[FigureRef]


class PDFParser(ABC):
    """Abstract base class for PDF parsing backends.

    Concrete implementations must override ``parse`` to convert raw PDF
    bytes into a structured ``ParsedPaper``.
    """

    @abstractmethod
    async def parse(self, pdf_bytes: bytes) -> ParsedPaper:
        """Parse raw PDF bytes into structured ParsedPaper."""
