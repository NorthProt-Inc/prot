from collections.abc import AsyncIterator

import httpx
from elevenlabs import AsyncElevenLabs

from prot.config import settings
from prot.log import get_logger

logger = get_logger(__name__)


class TTSClient:
    """Streaming TTS client using ElevenLabs Flash v2.5."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = AsyncElevenLabs(
            api_key=api_key or settings.elevenlabs_api_key,
        )
        self._cancelled = False

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM audio bytes for given text."""
        self._cancelled = False
        logger.info("Audio stream", text=text[:30], model=settings.elevenlabs_model)
        try:
            async for chunk in self._client.text_to_speech.stream(
                voice_id=settings.elevenlabs_voice_id,
                text=text,
                model_id=settings.elevenlabs_model,
                output_format=settings.elevenlabs_output_format,
            ):
                if self._cancelled:
                    break
                if isinstance(chunk, bytes):
                    yield chunk
        except (httpx.ConnectError, httpx.NetworkError, httpx.TimeoutException, OSError):
            logger.warning("TTS stream failed (network)", text=text[:30])
        except Exception:
            logger.exception("TTS stream failed", text=text[:30])

    async def warm(self) -> None:
        """Pre-warm HTTP connection pool by making a lightweight API call."""
        try:
            await self._client.voices.get_all()
            logger.info("TTS connection warmed")
        except Exception:
            logger.debug("TTS warm failed", exc_info=True)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    def flush(self) -> None:
        """Cancel current TTS stream."""
        self._cancelled = True
