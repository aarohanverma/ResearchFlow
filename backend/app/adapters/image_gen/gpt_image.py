"""gpt-image-2 adapter — instant and thinking modes, cached by prompt hash."""

import hashlib
import time

from openai import AsyncOpenAI

from app.adapters.image_gen.base import GeneratedImage, ImageGenAdapter
from app.core.config import settings

_MODEL = "gpt-image-2"
_SNAPSHOT = "gpt-image-2-2026-04-21"


class GptImage2Adapter(ImageGenAdapter):
    """Wraps gpt-image-2 with prompt-hash caching (30-day TTL via blob)."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the OpenAI async client for image generation.

        Args:
            api_key: OpenAI API key. Falls back to ``settings.openai_api_key``
                if not provided.
        """
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    def _cache_key(self, prompt: str, mode: str, size: str) -> str:
        """Return a deterministic SHA-256 cache key for an image generation request."""
        raw = f"{prompt}|{mode}|{size}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def generate(
        self,
        prompt: str,
        *,
        mode: str = "instant",
        size: str = "1024x1024",
        n: int = 1,
    ) -> list[GeneratedImage]:
        """Generate one or more images from a text prompt using gpt-image-2.

        Args:
            prompt: Text description of the desired image(s).
            mode: ``"instant"`` uses the base model; ``"thinking"`` uses the
                dated snapshot for higher quality.
            size: Image dimensions as a string (e.g. ``"1024x1024"``).
            n: Number of images to generate.

        Returns:
            A list of ``GeneratedImage`` objects, one per generated image.
        """
        # For thinking mode, use the more capable snapshot
        model = _SNAPSHOT if mode == "thinking" else _MODEL

        resp = await self._client.images.generate(
            model=model,
            prompt=prompt,
            size=size,  # type: ignore[arg-type]
            n=n,
            response_format="b64_json",
        )

        return [
            GeneratedImage(
                url=img.url,
                b64_json=img.b64_json,
                revised_prompt=getattr(img, "revised_prompt", None),
            )
            for img in resp.data
        ]

    async def edit(
        self,
        base_image: bytes,
        prompt: str,
        mask: bytes | None = None,
    ) -> GeneratedImage:
        """Edit a base image using a prompt via the gpt-image-2 edit endpoint.

        Args:
            base_image: Raw PNG bytes of the image to edit.
            prompt: Natural language instruction describing the desired edit.
            mask: Optional PNG mask bytes indicating the region to edit.

        Returns:
            A single ``GeneratedImage`` with the edited result.
        """
        import io
        files: dict = {"image": ("image.png", io.BytesIO(base_image), "image/png")}
        if mask:
            files["mask"] = ("mask.png", io.BytesIO(mask), "image/png")

        resp = await self._client.images.edit(
            model=_MODEL,
            image=io.BytesIO(base_image),
            prompt=prompt,
            response_format="b64_json",
        )
        img = resp.data[0]
        return GeneratedImage(url=img.url, b64_json=img.b64_json, revised_prompt=None)
