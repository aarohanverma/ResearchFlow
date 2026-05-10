"""Marp slide adapter — renders Marp markdown to HTML via marp-cli.

Marp (https://marp.app) converts Markdown with ``---`` separators into
beautiful HTML/PDF slide decks.  This adapter:
  1. Attempts to render via the ``marp`` CLI (npx or local install).
  2. Falls back to storing raw markdown if marp-cli is unavailable.

The HTML output is fully self-contained (inline CSS + JS) so it can be
served as a single file from BlobStorage.
"""

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path

from app.adapters.slides.base import SlidesAdapter, SlidesResult

log = logging.getLogger(__name__)

# ── Marp CLI availability check ───────────────────────────────────────────────

def _find_marp_cli() -> str | None:
    """Return the full path to marp-cli, or None if unavailable."""
    import os
    home = os.path.expanduser("~")

    # Prefer explicit full-path candidates so the binary works regardless
    # of whether the server process inherits ~/.npm-global/bin in its PATH.
    candidates = [
        os.path.join(home, ".npm-global", "bin", "marp"),
        os.path.join(home, ".local", "bin", "marp"),
        "/usr/local/bin/marp",
        "/usr/bin/marp",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            log.debug("marp-cli found at %s", path)
            return path

    # Fall back to PATH lookup
    found = shutil.which("marp") or shutil.which("npx")
    if found:
        log.debug("marp-cli found via PATH: %s", found)
    return found


_MARP_CMD = _find_marp_cli()


class MarpSlidesAdapter(SlidesAdapter):
    """Slide renderer backed by marp-cli.

    If marp-cli is not installed, ``render`` returns the raw Marp markdown
    with ``rendered_format="md"`` so the caller can still store and display
    the content (the frontend can render Marp markdown as a slide viewer).

    Args:
        allow_local_files: Pass ``--allow-local-files`` to marp-cli.
            Needed for local image references.
    """

    provider_id = "marp"
    supported_formats = ["html", "pdf", "md"]

    def __init__(self, allow_local_files: bool = False) -> None:
        """Initialise the Marp adapter.

        Args:
            allow_local_files: Pass ``--allow-local-files`` to marp-cli.
                Required when the Markdown references local image paths.
        """
        self._allow_local_files = allow_local_files

    async def render(self, markdown: str, *, output_format: str = "html") -> SlidesResult:
        """Render Marp markdown to HTML (preferred) or PDF.

        Falls back to ``"md"`` format if marp-cli is unavailable.

        Args:
            markdown: Fully-formed Marp markdown (includes front-matter).
            output_format: ``"html"`` (default) or ``"pdf"``.

        Returns:
            :class:`SlidesResult` with rendered bytes or raw markdown.
        """
        slide_count = self.count_slides(markdown)
        t0 = time.monotonic()

        if _MARP_CMD is None:
            log.info(
                "marp_slides.render: marp-cli not found — returning raw markdown "
                "(install with: npm install -g @marp-team/marp-cli)"
            )
            return SlidesResult(
                rendered_bytes=markdown.encode("utf-8"),
                rendered_format="md",
                markdown=markdown,
                slide_count=slide_count,
                provider_used=self.provider_id,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )

        rendered = await self._render_with_cli(markdown, output_format)
        latency_ms = int((time.monotonic() - t0) * 1000)

        if rendered is None:
            log.warning("marp_slides.render: CLI failed — falling back to markdown")
            return SlidesResult(
                rendered_bytes=markdown.encode("utf-8"),
                rendered_format="md",
                markdown=markdown,
                slide_count=slide_count,
                provider_used=self.provider_id,
                latency_ms=latency_ms,
            )

        log.info(
            "marp_slides.render complete slides=%d format=%s size=%d latency_ms=%d",
            slide_count, output_format, len(rendered), latency_ms,
        )
        return SlidesResult(
            rendered_bytes=rendered,
            rendered_format=output_format,
            markdown=markdown,
            slide_count=slide_count,
            provider_used=self.provider_id,
            latency_ms=latency_ms,
        )

    async def _render_with_cli(self, markdown: str, output_format: str) -> bytes | None:
        """Invoke marp-cli in a subprocess and return the rendered bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "slides.md"
            output_path = Path(tmpdir) / f"slides.{output_format}"
            input_path.write_text(markdown, encoding="utf-8")

            cmd: list[str]
            if _MARP_CMD and "npx" in (_MARP_CMD or ""):
                cmd = [_MARP_CMD, "@marp-team/marp-cli@latest"]
            else:
                cmd = [_MARP_CMD or "marp"]

            cmd += [str(input_path), "-o", str(output_path), "--html"]
            if output_format == "pdf":
                cmd += ["--pdf"]
            if self._allow_local_files:
                cmd += ["--allow-local-files"]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)

                if proc.returncode != 0:
                    log.warning(
                        "marp-cli exited %d stderr=%.300s",
                        proc.returncode, stderr.decode(errors="replace"),
                    )
                    return None

                if output_path.exists():
                    return output_path.read_bytes()

                log.warning("marp-cli succeeded but output file not found")
                return None

            except asyncio.TimeoutError:
                log.error("marp-cli timed out after 120s")
                return None
            except Exception as exc:
                log.error("marp-cli subprocess error: %s", exc)
                return None
