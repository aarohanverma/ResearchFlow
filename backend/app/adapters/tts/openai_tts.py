"""OpenAI TTS adapter — wraps the openai.audio.speech API.

Models: tts-1 (fast, lower quality) | tts-1-hd (higher quality, ~2× latency)
Voices: alloy, echo, fable, onyx, nova, shimmer

Usage in podcasts:
  HOST voice  → "alloy"  (neutral, clear, welcoming)
  EXPERT voice → "onyx"  (deep, authoritative)
  NEWCOMER guide → "nova" (warm, friendly)
"""

import logging
import time

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.adapters.tts.base import TTSAdapter, TTSResult
from app.core.config import settings

log = logging.getLogger(__name__)


def _is_retryable_tts(exc: Exception) -> bool:
    """Decide whether a TTS error is worth retrying.

    Retry on transient failures only (network, timeout, rate-limit, 5xx).
    Never retry on auth / bad-request / billing errors — those will repeat
    no matter how many times we retry.
    """
    import httpx

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    try:
        import openai  # local import to avoid hard-dep at module load
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
            return True
        if isinstance(exc, openai.APIConnectionError):
            return True
    except ImportError:
        pass
    return False

# ── Voice role constants ───────────────────────────────────────────────────────

HOST_VOICE = "alloy"
EXPERT_VOICE = "onyx"
GUIDE_VOICE = "nova"
NARRATOR_VOICE = "fable"

# Map expertise level → default guide voice
EXPERTISE_VOICES: dict[str, str] = {
    "newcomer": "nova",
    "practitioner": "alloy",
    "expert": "onyx",
}


class OpenAITTSAdapter(TTSAdapter):
    """TTS implementation backed by the OpenAI audio/speech endpoint.

    The adapter is stateless and thread-safe.  The OpenAI client is created
    once per instance and reused across calls.

    Args:
        model: TTS model to use (``"tts-1"`` or ``"tts-1-hd"``).
            Defaults to ``"tts-1-hd"`` for podcast quality.
        response_format: Output audio format.  ``"mp3"`` works across all
            browsers and devices; ``"opus"`` gives better compression.
    """

    provider_id = "openai"
    default_voice = HOST_VOICE
    supports_multiple_voices = True

    def __init__(
        self,
        model: str = "tts-1-hd",
        response_format: str = "mp3",
    ) -> None:
        """Initialise the adapter. The OpenAI client is created lazily on first call.

        Args:
            model: OpenAI TTS model identifier. ``"tts-1-hd"`` is higher
                quality; ``"tts-1"`` is faster and cheaper.
            response_format: Audio container format. ``"mp3"`` is the default
                for maximum browser/device compatibility.
        """
        self._model = model
        self._response_format = response_format
        self._client: object | None = None  # lazy-init

    @property
    def max_chars_per_call(self) -> int:
        """OpenAI TTS accepts up to 4096 characters per call."""
        return 4096

    def _get_client(self):
        """Lazily initialise the async OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesize ``text`` via OpenAI TTS and return raw audio bytes.

        Long texts are automatically chunked and the resulting MP3 segments
        are concatenated (MP3 byte concatenation is valid per the spec).

        Args:
            text: Plain text to convert.
            voice: One of ``alloy|echo|fable|onyx|nova|shimmer``.
                Defaults to ``self.default_voice``.
            model: Override the TTS model (``tts-1`` or ``tts-1-hd``).
            speed: Speech speed multiplier (0.25 – 4.0).

        Returns:
            :class:`TTSResult` with concatenated audio bytes.

        Raises:
            RuntimeError: If the API key is missing or the API call fails.
        """
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for TTS generation."
            )

        selected_voice = voice or self.default_voice
        selected_model = model or self._model
        client = self._get_client()

        chunks = self.chunk_text(text)
        audio_parts: list[bytes] = []
        total_chars = 0
        t0 = time.monotonic()

        for chunk in chunks:
            if not chunk.strip():
                continue

            async def _call_once():
                """Single OpenAI TTS request — retried on transient errors."""
                return await client.audio.speech.create(
                    model=selected_model,
                    voice=selected_voice,
                    input=chunk,
                    response_format=self._response_format,
                    speed=speed,
                )

            # 3 attempts with jittered exponential backoff (1s → up to 8s)
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception(_is_retryable_tts),
                    stop=stop_after_attempt(3),
                    wait=wait_exponential_jitter(initial=1, max=8),
                    reraise=True,
                ):
                    with attempt:
                        response = await _call_once()
                audio_parts.append(response.content)
                total_chars += len(chunk)
                log.debug(
                    "openai_tts.synthesize voice=%s chars=%d",
                    selected_voice, len(chunk),
                )
            except Exception as exc:
                log.error("openai_tts.synthesize failed chunk=%d err=%s", len(chunk), exc)
                raise

        latency_ms = int((time.monotonic() - t0) * 1000)
        combined = b"".join(audio_parts)

        return TTSResult(
            audio_bytes=combined,
            audio_format=self._response_format,
            provider_used=self.provider_id,
            model_used=selected_model,
            character_count=total_chars,
            latency_ms=latency_ms,
        )

    async def synthesize_dialogue(
        self,
        utterances: list[dict],
    ) -> bytes:
        """Synthesize a multi-speaker dialogue and return concatenated audio.

        Args:
            utterances: List of ``{speaker, text, voice}`` dicts.  The
                ``voice`` key overrides the per-speaker default.

        Returns:
            Concatenated MP3 bytes for the full dialogue.
        """
        audio_parts: list[bytes] = []
        for utt in utterances:
            text = utt.get("text", "").strip()
            if not text:
                continue
            voice = utt.get("voice") or self.default_voice
            result = await self.synthesize(text, voice=voice)
            audio_parts.append(result.audio_bytes)
        return b"".join(audio_parts)
