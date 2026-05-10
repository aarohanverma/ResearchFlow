"""Slides adapter ABC — every slide-generation backend must conform.

Implementations take a Marp-compatible markdown string and produce either
a rendered artifact (HTML, PDF) or simply validate the markdown.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SlidesResult:
    """Result of a slide rendering operation.

    Attributes:
        rendered_bytes: The rendered output (HTML bytes, PDF bytes, etc.).
            ``None`` if only markdown storage is supported.
        rendered_format: File extension of the rendered output (``"html"``,
            ``"pdf"``).  ``"md"`` when no rendering occurred.
        markdown: The Marp markdown source that was rendered.
        slide_count: Number of slides extracted from the markdown.
        provider_used: Identifier of the backend (e.g. ``"marp"``).
        latency_ms: Wall-clock rendering time in milliseconds.
    """

    rendered_bytes: bytes | None
    rendered_format: str
    markdown: str
    slide_count: int
    provider_used: str
    latency_ms: int


class SlidesAdapter(ABC):
    """Abstract base class for slide-generation backends.

    Concrete implementations are responsible for taking a fully-formed
    Marp markdown string and producing a rendered artifact.  The LLM
    prompt and markdown generation happen in the workflow layer, not here.

    Attributes:
        provider_id: Short identifier (e.g. ``"marp"``).
        supported_formats: List of output formats this backend can produce.
    """

    provider_id: str
    supported_formats: list[str]

    @abstractmethod
    async def render(self, markdown: str, *, output_format: str = "html") -> SlidesResult:
        """Render Marp markdown into a slide artifact.

        Args:
            markdown: Full Marp-compatible markdown string.
            output_format: Desired output format.  Falls back to ``"md"``
                if the backend does not support the requested format.

        Returns:
            A :class:`SlidesResult` with the rendered output and metadata.
        """

    @staticmethod
    def count_slides(markdown: str) -> int:
        """Count the number of slides in a Marp markdown string.

        Slides are separated by ``---`` on its own line.

        Args:
            markdown: Marp markdown string.

        Returns:
            Number of slides (minimum 1).
        """
        import re
        return max(1, len(re.split(r"^---\s*$", markdown, flags=re.MULTILINE)))
