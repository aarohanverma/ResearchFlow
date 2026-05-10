"""PDF parser ABC and canonical output schema.

All parser backends must produce a ``ParsedPaper`` from raw PDF bytes.
Downstream code (workflows, repositories) must depend only on these types —
never on parser-specific objects.

Schema is backward-compatible: all new fields carry defaults so existing
callers that only read ``sections`` continue to work unchanged.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── Primitive document elements ───────────────────────────────────────────────

@dataclass
class Reference:
    """A bibliographic reference extracted from a paper.

    Attributes:
        title: Reference title string (may be empty if only an arXiv ID was found).
        authors: List of author name strings.
        arxiv_id: arXiv paper identifier if present (e.g. ``"2301.07041"``).
        year: Publication year as a string, if extractable.
        venue: Journal or conference name, if present.
    """

    title: str
    authors: list[str] = field(default_factory=list)
    arxiv_id: str | None = None
    year: str | None = None
    venue: str | None = None


@dataclass
class FigureRef:
    """Metadata for a figure extracted from a PDF page.

    Attributes:
        caption: Figure caption text.
        page_no: Zero-based page index where the figure appears.
        bbox: Bounding box coordinates ``[x0, y0, x1, y1]``.
        image_bytes: Raw PNG/JPEG bytes of the figure, if extracted.
        figure_id: Canonical identifier (e.g. ``"Figure 3"``).
    """

    caption: str
    page_no: int
    bbox: list[float] = field(default_factory=list)
    image_bytes: bytes | None = None
    figure_id: str | None = None


@dataclass
class TableData:
    """A table extracted from the document.

    Attributes:
        markdown: Table content serialized as Markdown (``|`` syntax).
        caption: Table caption, if present.
        page_no: Zero-based page index.
        table_id: Canonical identifier (e.g. ``"Table 2"``).
    """

    markdown: str
    caption: str | None = None
    page_no: int | None = None
    table_id: str | None = None


@dataclass
class EquationData:
    """A mathematical equation or expression found in the document.

    Attributes:
        source: Raw LaTeX or plain-text representation.
        inline: ``True`` when the equation appears inline in a sentence.
        label: Equation label or number (e.g. ``"(3)"``), if present.
    """

    source: str
    inline: bool = False
    label: str | None = None


# ── Section ───────────────────────────────────────────────────────────────────

@dataclass
class Section:
    """A logical section of a parsed academic paper.

    Backward-compatible: ``section_type`` and ``content`` retain the same
    semantics as before; new fields all default to empty / ``None``.

    Attributes:
        section_type: Canonical section label (e.g. ``"abstract"``,
            ``"introduction"``, ``"methodology"``, ``"results"``).
        content: Full text of the section body.
        heading: Raw heading string as it appears in the document.
        level: Hierarchy depth (1 = top-level, 2 = subsection, …).
        math_blocks: LaTeX math expressions found in the section.
        tables: Raw table strings extracted from the section.
        figures: Figure references found within this section.
        equations: Structured equation objects from this section.
        subsections: Nested child sections (populated by structured parsers).
    """

    # ── Existing fields (stable contract) ────────────────────────────────────
    section_type: str
    content: str
    math_blocks: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    # ── Enhanced fields (new — all default to safe values) ───────────────────
    heading: str | None = None
    level: int = 1
    figures: list[FigureRef] = field(default_factory=list)
    equations: list[EquationData] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)


# ── ParsedPaper ───────────────────────────────────────────────────────────────

@dataclass
class ParsedPaper:
    """Structured representation of a fully parsed academic paper.

    Backward-compatible: ``title``, ``sections``, ``references``, and
    ``figures`` retain their original semantics.  New fields carry defaults.

    Attributes:
        title: Paper title string.
        sections: Ordered list of extracted sections.
        references: List of bibliographic references.
        figures: All figure metadata objects across the full document.
        tables: All tables across the full document (structured).
        equations: All equations across the full document (structured).
        abstract: Convenience field — first abstract section content.
        document_metadata: Parser-extracted document properties (DOI, venue, …).
        parser_name: Identifier of the parser that produced this object.
        fallback_used: ``True`` if a fallback parser was invoked.
        parse_duration_ms: Wall-clock parse time in milliseconds.
        parser_confidence: Heuristic confidence score in [0, 1].
    """

    # ── Existing fields (stable contract) ────────────────────────────────────
    title: str
    sections: list[Section]
    references: list[Reference]
    figures: list[FigureRef]

    # ── Enhanced fields (new — all default to safe values) ───────────────────
    tables: list[TableData] = field(default_factory=list)
    equations: list[EquationData] = field(default_factory=list)
    abstract: str | None = None
    document_metadata: dict = field(default_factory=dict)
    parser_name: str = "unknown"
    fallback_used: bool = False
    parse_duration_ms: int = 0
    parser_confidence: float = 1.0


# ── ABC ───────────────────────────────────────────────────────────────────────

class PDFParser(ABC):
    """Abstract base class for PDF parsing backends.

    Each concrete implementation must:
    1. Override ``parse`` to convert raw PDF bytes into a ``ParsedPaper``.
    2. Set ``parser_name`` on the returned object.
    3. Never raise from within ``parse`` — return a best-effort result
       and let the caller decide whether to invoke a fallback.

    The ABC deliberately keeps no state so implementations can be
    instantiated once and reused across concurrent requests.
    """

    @abstractmethod
    async def parse(self, pdf_bytes: bytes) -> ParsedPaper:
        """Parse raw PDF bytes into a structured :class:`ParsedPaper`.

        Args:
            pdf_bytes: Raw bytes of the PDF file to parse.

        Returns:
            A ``ParsedPaper`` populated with as much structure as the
            backend can reliably extract.
        """
