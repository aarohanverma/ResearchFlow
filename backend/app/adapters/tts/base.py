"""TTS adapter ABC — every TTS backend must conform to this contract.

Backends are stateless; instantiate once, call many times concurrently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TTSResult:
    """Result of a single TTS synthesis call.

    Attributes:
        audio_bytes: Raw audio data in the target format.
        audio_format: File format identifier (e.g. ``"mp3"``, ``"wav"``).
        provider_used: Identifier of the backend that produced this audio.
        model_used: Specific model string.
        character_count: Number of input characters converted.
        latency_ms: Wall-clock time of the API call in milliseconds.
    """

    audio_bytes: bytes
    audio_format: str
    provider_used: str
    model_used: str
    character_count: int
    latency_ms: int


class TTSAdapter(ABC):
    """Abstract base class for text-to-speech backends.

    Voice names are backend-specific; callers should use the constants
    provided by each concrete implementation rather than hardcoding strings.

    Attributes:
        provider_id: Short identifier (e.g. ``"openai"``).
        default_voice: Voice used when none is specified.
        supports_multiple_voices: ``True`` if the backend can produce
            distinct voices in the same session.
    """

    provider_id: str
    default_voice: str
    supports_multiple_voices: bool = True

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Convert ``text`` to audio bytes.

        Args:
            text: Plain text to convert (no SSML unless backend supports it).
            voice: Backend-specific voice identifier.  Falls back to
                ``default_voice`` when ``None``.
            model: Backend-specific model override.
            speed: Playback speed multiplier (1.0 = normal).

        Returns:
            A :class:`TTSResult` with the raw audio bytes.
        """

    @property
    @abstractmethod
    def max_chars_per_call(self) -> int:
        """Maximum number of characters accepted in a single synthesis call."""

    def chunk_text(self, text: str, max_chars: int | None = None) -> list[str]:
        """Split ``text`` into chunks that fit within ``max_chars_per_call``.

        Splits on sentence boundaries where possible to avoid cutting words.
        Useful for podcasts where the full script exceeds TTS limits.

        Args:
            text: The full text to split.
            max_chars: Override the default limit.

        Returns:
            List of non-overlapping text chunks.
        """
        limit = max_chars or self.max_chars_per_call
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        current_start = 0
        while current_start < len(text):
            end = current_start + limit
            if end >= len(text):
                chunks.append(text[current_start:].strip())
                break

            # Try to split on a sentence boundary within the last 20% of the chunk
            search_start = end - limit // 5
            boundary = -1
            for sep in (". ", "! ", "? ", "\n\n", "\n"):
                idx = text.rfind(sep, search_start, end)
                if idx != -1:
                    boundary = idx + len(sep)
                    break

            if boundary == -1:
                boundary = end

            chunk = text[current_start:boundary].strip()
            if chunk:
                chunks.append(chunk)
            current_start = boundary

        return [c for c in chunks if c]
