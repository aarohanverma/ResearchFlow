"""Docling PDF parser — IBM's structured document intelligence library.

Docling (https://github.com/DS4SD/docling) excels at:
  - Section hierarchy preservation
  - Table extraction (structured, not just raw text)
  - Equation / formula detection
  - Figure-caption alignment
  - Multi-column academic layout handling
  - Citation / reference extraction
  - Deterministic, CPU-efficient output

Falls back gracefully when docling is not installed.

SECURITY: PDF bytes are treated as untrusted external data.
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from app.adapters.pdf.base import (
    EquationData,
    FigureRef,
    ParsedPaper,
    PDFParser,
    Reference,
    Section,
    TableData,
)

log = logging.getLogger(__name__)

# ── Availability check — optional dependency ──────────────────────────────────

try:
    from docling.document_converter import DocumentConverter  # noqa: F401
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False
    log.debug(
        "docling not installed — parser chain will use Marker → Gemini Vision. "
        "To enable Docling: pip install docling && set PDF_PARSER=docling"
    )

# OCR is OFF by default. Even though EasyOCR runs on CPU, downloading and
# loading its detection + recognition models can spike RAM by ~2 GB on
# first call — enough to crash WSL or a small Docker VM. Set
# DOCLING_OCR=1 to enable it explicitly when you have spare RAM.
import os
_DOCLING_OCR_ENABLED = os.environ.get("DOCLING_OCR", "").lower() in ("1", "true", "yes", "on")

try:
    import easyocr  # noqa: F401
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False


# ── Section keyword mapping (mirrors MarkerParser for consistency) ────────────

_SECTION_KEYWORDS: dict[str, str] = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "preliminaries": "background",
    "notation": "background",
    "method": "methodology",
    "methodology": "methodology",
    "approach": "methodology",
    "model": "methodology",
    "architecture": "methodology",
    "framework": "methodology",
    "system": "methodology",
    "experiment": "results",
    "result": "results",
    "evaluation": "results",
    "benchmark": "results",
    "ablation": "results",
    "analysis": "discussion",
    "discussion": "discussion",
    "limitation": "limitations",
    "future work": "future_work",
    "conclusion": "conclusion",
    "appendix": "appendix",
    "supplementary": "appendix",
    "reference": "references",
    "bibliography": "references",
    "acknowledgement": "acknowledgements",
    "acknowledgment": "acknowledgements",
}


def _classify_heading(heading: str) -> str:
    """Map a section heading to a canonical section-type label."""
    lower = heading.lower()
    for keyword, label in _SECTION_KEYWORDS.items():
        if keyword in lower:
            return label
    return "other"


class DoclingParser(PDFParser):
    """Structured PDF parser backed by IBM Docling.

    Extracts a rich ``ParsedPaper`` including:
    - Hierarchical sections with heading levels
    - Structured tables as Markdown
    - Equations with LaTeX source
    - Figures with captions
    - Bibliographic references

    Docling's ``DocumentConverter`` is CPU-only and thread-safe; the
    synchronous conversion is offloaded to a thread-pool executor.
    """

    # Maximum wall-clock seconds to allow for a single PDF parse.
    # Docling's CPU path can be slow on large academic PDFs; 300 s is generous
    # while still preventing a stalled thread from blocking the executor pool
    # indefinitely.
    _PARSE_TIMEOUT_S: int = 300

    async def parse(self, pdf_bytes: bytes) -> ParsedPaper:
        """Parse PDF bytes via Docling asynchronously.

        Delegates the synchronous Docling conversion to a thread-pool
        executor to avoid blocking the async event loop.  A hard timeout of
        :attr:`_PARSE_TIMEOUT_S` seconds is applied so a stalled parse cannot
        hold the executor thread pool indefinitely.

        Args:
            pdf_bytes: Raw bytes of the PDF to parse.

        Returns:
            A richly structured ``ParsedPaper``.

        Raises:
            RuntimeError: If docling is not installed.
            asyncio.TimeoutError: If parsing exceeds :attr:`_PARSE_TIMEOUT_S`.
        """
        if not _DOCLING_AVAILABLE:
            raise RuntimeError(
                "docling is not installed. "
                "Add 'docling' to requirements.txt and rebuild."
            )
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._parse_sync, pdf_bytes),
                timeout=self._PARSE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.error(
                "docling.parse: timed out after %ds — PDF may be too large or corrupt",
                self._PARSE_TIMEOUT_S,
            )
            raise

    def _parse_sync(self, pdf_bytes: bytes) -> ParsedPaper:
        """Synchronous Docling conversion — runs in a thread-pool.

        Hardware selection:
            * If a working CUDA-capable PyTorch is detected, use the GPU.
            * Otherwise, fall back to CPU. Either path is fully functional;
              CPU is the safe default on most laptops and CI machines.

        OCR backend:
            * EasyOCR is enabled when installed (CPU or GPU); it installs
              purely from pip and is the most portable option.
            * If EasyOCR is missing, Docling parses without OCR and the
              fallback chain handles scanned PDFs (Marker → Gemini Vision).
        """
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        t0 = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        # Detect a usable GPU. Any failure → CPU. We import torch lazily so a
        # missing torch never crashes the parser.
        gpu_available = False
        try:
            import torch  # type: ignore[import-not-found]
            gpu_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        except Exception:
            gpu_available = False

        try:
            # Enable OCR only when both DOCLING_OCR=1 is set AND EasyOCR is
            # importable. Loading EasyOCR models can spike RAM by ~2 GB on
            # first call, so we keep this opt-in to protect small VMs.
            ocr_on = _DOCLING_OCR_ENABLED and _EASYOCR_AVAILABLE
            pipeline_options = PdfPipelineOptions(do_ocr=ocr_on)
            if ocr_on:
                try:
                    from docling.datamodel.pipeline_options import EasyOcrOptions
                    pipeline_options.ocr_options = EasyOcrOptions(use_gpu=gpu_available)
                except Exception:
                    pass

            # Pin Docling to GPU/CPU explicitly when the AcceleratorOptions
            # API is available. Tolerated if absent across Docling releases.
            try:
                from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
                pipeline_options.accelerator_options = AcceleratorOptions(
                    device=AcceleratorDevice.CUDA if gpu_available else AcceleratorDevice.CPU,
                )
            except Exception:
                pass

            log.debug(
                "docling._parse_sync device=%s ocr=%s",
                "cuda" if gpu_available else "cpu",
                ocr_on,
            )

            try:
                from docling.document_converter import PdfFormatOption
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    }
                )
            except Exception:
                # Older Docling versions accept pipeline_options directly
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: pipeline_options}
                )
            result = converter.convert(str(tmp_path))
            doc = result.document
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        duration_ms = int((time.monotonic() - t0) * 1000)

        sections = self._extract_sections(doc)
        references = self._extract_references(doc)
        figures = self._extract_figures(doc)
        tables = self._extract_tables(doc)
        equations = self._extract_equations(doc)
        title = self._extract_title(doc, sections)
        abstract = self._extract_abstract(sections)
        metadata = self._extract_metadata(doc)

        log.info(
            "docling.parse complete title=%.60s sections=%d tables=%d "
            "equations=%d refs=%d duration_ms=%d",
            title, len(sections), len(tables), len(equations),
            len(references), duration_ms,
        )

        return ParsedPaper(
            title=title,
            sections=sections,
            references=references,
            figures=figures,
            tables=tables,
            equations=equations,
            abstract=abstract,
            document_metadata=metadata,
            parser_name="docling",
            fallback_used=False,
            parse_duration_ms=duration_ms,
            parser_confidence=0.90,
        )

    # ── Private extraction helpers ─────────────────────────────────────────────

    def _extract_title(self, doc: object, sections: list[Section]) -> str:
        """Extract document title from Docling metadata or first section heading."""
        try:
            if hasattr(doc, "name") and doc.name:
                return str(doc.name)
        except Exception:
            pass
        # Fallback: first section content up to 120 chars
        if sections:
            return sections[0].content[:120].strip()
        return "Untitled"

    def _extract_sections(self, doc: object) -> list[Section]:
        """Convert Docling document texts into canonical Section objects."""
        sections: list[Section] = []
        current_heading = "body"
        current_level = 1
        current_content_parts: list[str] = []

        def _flush() -> None:
            if current_content_parts:
                body = "\n\n".join(current_content_parts).strip()
                if body:
                    section_type = _classify_heading(current_heading)
                    import re
                    math_blocks = re.findall(r"\$\$(.+?)\$\$", body, re.DOTALL)
                    sections.append(Section(
                        section_type=section_type,
                        content=body,
                        heading=current_heading,
                        level=current_level,
                        math_blocks=math_blocks,
                    ))
            current_content_parts.clear()

        try:
            for text_item in doc.texts:
                label = getattr(text_item, "label", None)
                text = getattr(text_item, "text", "") or ""
                if not text.strip():
                    continue

                # Section headings
                if label in ("section_header", "title") or (
                    hasattr(text_item, "label")
                    and str(label).lower() in ("section_header",)
                ):
                    _flush()
                    current_heading = text.strip()
                    current_level = getattr(text_item, "level", 1) or 1
                else:
                    current_content_parts.append(text)

            _flush()
        except Exception as exc:
            log.warning("docling._extract_sections error: %s", exc)
            # Best-effort: return whatever we have
            if not sections:
                # Export full markdown as single section
                try:
                    md = doc.export_to_markdown()
                    sections.append(Section(
                        section_type="body",
                        content=md,
                        heading="Full Text",
                    ))
                except Exception:
                    pass

        return sections or [Section(section_type="body", content="(parse error)")]

    def _extract_references(self, doc: object) -> list[Reference]:
        """Extract bibliographic references from Docling document."""
        refs: list[Reference] = []
        import re

        try:
            for text_item in doc.texts:
                label = getattr(text_item, "label", None)
                if str(label).lower() in ("reference", "bibliography"):
                    text = getattr(text_item, "text", "") or ""
                    arxiv_ids = re.findall(r"arXiv:(\d{4}\.\d{4,5})", text)
                    for aid in arxiv_ids:
                        refs.append(Reference(title="", arxiv_id=aid))
        except Exception as exc:
            log.debug("docling._extract_references error: %s", exc)

        return refs

    def _extract_figures(self, doc: object) -> list[FigureRef]:
        """Extract figure metadata from Docling pictures."""
        figures: list[FigureRef] = []
        try:
            for pic in doc.pictures:
                caption = ""
                try:
                    if hasattr(pic, "caption_text"):
                        caption = pic.caption_text(doc) or ""
                    elif hasattr(pic, "caption"):
                        caption = str(pic.caption) if pic.caption else ""
                except Exception:
                    pass
                page = getattr(pic, "prov", [{}])[0].get("page", 0) if hasattr(pic, "prov") else 0
                figures.append(FigureRef(caption=caption, page_no=page))
        except Exception as exc:
            log.debug("docling._extract_figures error: %s", exc)
        return figures

    def _extract_tables(self, doc: object) -> list[TableData]:
        """Extract tables as Markdown from Docling table items."""
        tables: list[TableData] = []
        try:
            for i, table in enumerate(doc.tables):
                try:
                    markdown = table.export_to_markdown()
                except Exception:
                    markdown = str(table)
                caption = ""
                try:
                    if hasattr(table, "caption_text"):
                        caption = table.caption_text(doc) or ""
                except Exception:
                    pass
                page = None
                if hasattr(table, "prov") and table.prov:
                    page = table.prov[0].get("page")
                tables.append(TableData(
                    markdown=markdown,
                    caption=caption or None,
                    page_no=page,
                    table_id=f"Table {i + 1}",
                ))
        except Exception as exc:
            log.debug("docling._extract_tables error: %s", exc)
        return tables

    def _extract_equations(self, doc: object) -> list[EquationData]:
        """Extract equations and formulae from Docling formula items."""
        equations: list[EquationData] = []
        try:
            formula_items = getattr(doc, "equations", None) or []
            for eq in formula_items:
                source = getattr(eq, "text", "") or ""
                if source:
                    equations.append(EquationData(source=source, inline=False))
        except Exception as exc:
            log.debug("docling._extract_equations error: %s", exc)
        return equations

    def _extract_metadata(self, doc: object) -> dict:
        """Extract document-level metadata from Docling."""
        meta: dict = {}
        try:
            if hasattr(doc, "description"):
                d = doc.description
                if hasattr(d, "title") and d.title:
                    meta["title"] = d.title
                if hasattr(d, "authors") and d.authors:
                    meta["authors"] = d.authors
                if hasattr(d, "language") and d.language:
                    meta["language"] = d.language
        except Exception:
            pass
        return meta

    def _extract_abstract(self, sections: list[Section]) -> str | None:
        """Return abstract content from the first 'abstract' section."""
        for s in sections:
            if s.section_type == "abstract":
                return s.content
        return None
