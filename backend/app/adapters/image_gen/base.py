"""ImageGenAdapter ABC — gpt-image-2 is the default implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GeneratedImage:
    """Result of a single image generation or edit request.

    Attributes:
        url: Temporary CDN URL returned by the API, if available.
        b64_json: Base-64 encoded image data, if returned in that format.
        revised_prompt: The prompt as revised by the model, if applicable.
        blob_path: Canonical blob storage path set after upload to
            persistent blob storage; ``None`` until uploaded.
    """

    url: str | None
    b64_json: str | None
    revised_prompt: str | None
    blob_path: str | None = None   # set after upload to blob storage


class ImageGenAdapter(ABC):
    """Abstract base class for image generation backends.

    Subclasses must implement ``generate``. The default ``edit``
    implementation raises ``NotImplementedError``.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        mode: str = "instant",   # instant | thinking
        size: str = "1024x1024",
        n: int = 1,
    ) -> list[GeneratedImage]:
        """Generate images from a text prompt."""

    async def edit(
        self,
        base_image: bytes,
        prompt: str,
        mask: bytes | None = None,
    ) -> GeneratedImage:
        """Edit a base image using a prompt and optional mask.

        Args:
            base_image: Raw bytes of the image to edit.
            prompt: Natural language instruction describing the desired edit.
            mask: Optional mask bytes indicating the region to edit.

        Raises:
            NotImplementedError: This adapter does not support image editing.
        """
        raise NotImplementedError("Edit not supported by this adapter")
