"""ElevenLabs Scribe v2 Realtime WebSocket streaming STT client."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from prot.config import settings
from prot.log import get_logger

logger = get_logger(__name__)

_WS_BASE = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


class STTClient:
    """ElevenLabs Scribe v2 Realtime WebSocket streaming client."""

    def __init__(
        self,
        api_key: str | None = None,
        on_transcript: Callable[[str, bool], Awaitable[None]] | None = None,
        on_utterance_end: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._api_key = api_key or settings.elevenlabs_api_key
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._ws: websockets.ClientConnection | None = None
        self._recv_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Open WebSocket connection to ElevenLabs and start receiving."""
        await self.disconnect()
        try:
            url = (
                f"{_WS_BASE}"
                f"?model_id=scribe_v2_realtime"
                f"&language_code={settings.stt_language}"
                f"&audio_format=pcm_{settings.sample_rate}"
                f"&commit_strategy=vad"
            )
            headers = {"xi-api-key": self._api_key}
            self._ws = await websockets.connect(url, additional_headers=headers)

            # Wait for session_started
            raw = await self._ws.recv()
            session = json.loads(raw)
            if session.get("message_type") != "session_started":
                logger.warning("Unexpected first message", msg=session)

            logger.info("Connected", model="scribe_v2_realtime", lang=settings.stt_language)
            self._recv_task = asyncio.create_task(self._recv_loop())
        except Exception:
            logger.exception("STT connect failed")
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
            self._ws = None

    async def send_audio(self, data: bytes) -> None:
        """Send PCM audio chunk to ElevenLabs."""
        if self._ws is None:
            return
        try:
            msg = json.dumps({
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(data).decode(),
                "commit": False,
                "sample_rate": settings.sample_rate,
            })
            await self._ws.send(msg)
        except Exception:
            logger.warning("Send failed, disconnecting")
            try:
                await self.disconnect()
            except Exception:
                self._ws = None
                self._recv_task = None

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _recv_loop(self) -> None:
        """Receive and dispatch messages from ElevenLabs."""
        ws = self._ws
        while True:
            try:
                raw = await ws.recv()
            except (ConnectionClosedOK, ConnectionClosedError):
                logger.debug("STT websocket closed")
                break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("STT recv error")
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message from STT")
                continue

            msg_type = msg.get("message_type", "")

            if msg_type == "partial_transcript":
                text = msg.get("text", "")
                if text:
                    await self._fire_transcript(text, is_final=False)

            elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
                text = msg.get("text", "")
                if text:
                    await self._fire_transcript(text, is_final=True)
                    await self._fire_utterance_end()

            elif msg_type in ("error", "auth_error", "input_error"):
                logger.error("STT error", error=msg.get("error", ""))

    async def _fire_transcript(self, text: str, is_final: bool) -> None:
        """Invoke the transcript callback if registered."""
        if self._on_transcript:
            await self._on_transcript(text, is_final)

    async def _fire_utterance_end(self) -> None:
        """Invoke the utterance end callback if registered."""
        if self._on_utterance_end:
            await self._on_utterance_end()
