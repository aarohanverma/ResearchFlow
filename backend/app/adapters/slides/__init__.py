"""Slides adapter package — factory and public re-exports."""

from app.adapters.slides.base import SlidesAdapter, SlidesResult
from app.adapters.slides.marp import MarpSlidesAdapter


def get_slides_adapter() -> SlidesAdapter:
    """Return the configured slides-generation backend.

    Currently only Marp is implemented.  Future backends (reveal.js,
    Impress.js, Google Slides API) can be registered here without
    changing callers.

    Returns:
        Configured :class:`SlidesAdapter` instance.
    """
    return MarpSlidesAdapter()


__all__ = ["SlidesAdapter", "SlidesResult", "MarpSlidesAdapter", "get_slides_adapter"]
