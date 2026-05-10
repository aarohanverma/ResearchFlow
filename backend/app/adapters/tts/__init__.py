"""TTS adapter package — factory and public re-exports."""

from app.adapters.tts.base import TTSAdapter, TTSResult
from app.adapters.tts.openai_tts import (
    EXPERT_VOICE,
    EXPERTISE_VOICES,
    GUIDE_VOICE,
    HOST_VOICE,
    NARRATOR_VOICE,
    OpenAITTSAdapter,
)
from app.core.config import settings

# Module-level singleton — avoids rebuilding the adapter (and its lazy client)
# on every podcast segment synthesis call.
_tts_singleton: TTSAdapter | None = None


def get_tts_adapter() -> TTSAdapter:
    """Return the configured TTS backend singleton.

    The instance is created once and reused across all calls so the
    underlying async OpenAI client is shared and its connection pool
    is not recreated for every synthesis request.

    Currently only OpenAI TTS is implemented.  The factory is kept separate
    so additional backends (ElevenLabs, Google TTS, Azure TTS) can be
    registered without changing callers.

    Returns:
        Configured :class:`TTSAdapter` instance.
    """
    global _tts_singleton
    if _tts_singleton is not None:
        return _tts_singleton

    provider = getattr(settings, "tts_provider", "openai").lower()

    if provider == "openai":
        model = getattr(settings, "tts_model", "tts-1-hd")
        _tts_singleton = OpenAITTSAdapter(model=model)
    else:
        # Default fallback — always return OpenAI if provider unknown
        _tts_singleton = OpenAITTSAdapter()

    return _tts_singleton


__all__ = [
    "TTSAdapter",
    "TTSResult",
    "OpenAITTSAdapter",
    "get_tts_adapter",
    "HOST_VOICE",
    "EXPERT_VOICE",
    "GUIDE_VOICE",
    "NARRATOR_VOICE",
    "EXPERTISE_VOICES",
]
