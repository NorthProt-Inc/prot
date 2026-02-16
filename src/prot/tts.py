from collections.abc import AsyncIterator

from elevenlabs import AsyncElevenLabs

from prot.config import settings


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

    def flush(self) -> None:
        """Cancel current TTS stream."""
        self._cancelled = True
