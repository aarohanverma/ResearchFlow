from app.adapters.image_gen.base import GeneratedImage, ImageGenAdapter
from app.adapters.image_gen.gpt_image import GptImage2Adapter
from app.core.config import settings


def get_image_gen_adapter() -> ImageGenAdapter:
    """Return the configured image generation adapter.

    Returns:
        A ``GptImage2Adapter`` instance using the settings-configured API key.
    """
    return GptImage2Adapter()


__all__ = ["ImageGenAdapter", "GeneratedImage", "get_image_gen_adapter"]
