"""Deepgram Nova-3 WebSocket streaming STT client."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from deepgram import AsyncDeepgramClient
from deepgram.extensions.types.sockets.listen_v1_results_event import (
    ListenV1ResultsEvent,
)
from deepgram.extensions.types.sockets.listen_v1_utterance_end_event import (
    ListenV1UtteranceEndEvent,
)
from websockets.exceptions import ConnectionClosedOK

from prot.config import settings
from prot.log import get_logger

logger = get_logger(__name__)


class STTClient:
    """Deepgram Nova-3 WebSocket streaming client."""

    def __init__(
        self,
        api_key: str | None = None,
        on_transcript: Callable[[str, bool], Awaitable[None]] | None = None,
        on_utterance_end: Callable[[], Awaitable[None]] | None = None,
        keyterms: list[str] | None = None,
    ) -> None:
        self._client = AsyncDeepgramClient(
            api_key=api_key or settings.deepgram_api_key,
        )
        self._connection: Any = None
        self._connection_ctx: Any = None
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._keyterms = keyterms or []
        self._recv_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Open WebSocket connection to Deepgram and start receiving."""
        await self.disconnect()
        try:
            self._connection_ctx = self._client.listen.v1.connect(
                model=settings.deepgram_model,
                language=settings.deepgram_language,
                encoding="linear16",
                sample_rate=str(settings.sample_rate),
                channels="1",
                smart_format="true",
                interim_results="true",
                utterance_end_ms=str(settings.deepgram_utterance_end_ms),
                endpointing=str(settings.deepgram_endpointing),
                keyterm=self._keyterms if self._keyterms else None,
            )
            self._connection = await self._connection_ctx.__aenter__()
            logger.info("Connected", model=settings.deepgram_model, lang=settings.deepgram_language)
            self._recv_task = asyncio.create_task(self._recv_loop())
        except Exception:
            logger.exception("STT connect failed")
            self._connection = None
            self._connection_ctx = None

    async def send_audio(self, data: bytes) -> None:
        """Send PCM audio chunk to Deepgram."""
        if self._connection:
            try:
                await self._connection.send_media(data)
            except Exception:
                logger.warning("Send failed, disconnecting")
                try:
                    await self.disconnect()
                except Exception:
                    self._connection = None
                    self._connection_ctx = None

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        if self._connection_ctx:
            await self._connection_ctx.__aexit__(None, None, None)
            self._connection = None
            self._connection_ctx = None

    async def _recv_loop(self) -> None:
        """Receive and dispatch messages from Deepgram."""
        conn = self._connection
        while conn:
            try:
                result = await conn.recv()
            except ConnectionClosedOK:
                logger.debug("STT websocket closed normally")
                break
            except Exception:
                logger.exception("STT recv error")
                break
            if isinstance(result, ListenV1ResultsEvent):
                await self._on_message(result)
            elif isinstance(result, ListenV1UtteranceEndEvent):
                await self._on_utt_end(result)

    async def _on_message(self, result: ListenV1ResultsEvent) -> None:
        """Handle a transcript result event."""
        alt = result.channel.alternatives[0]
        if not alt.transcript:
            return
        # Reconstruct from words array for proper Korean spacing
        if alt.words:
            transcript = " ".join(
                w.punctuated_word or w.word for w in alt.words
            )
            transcript = " ".join(transcript.split())  # normalize spaces
            logger.debug(
                "STT words",
                raw=alt.transcript[:80],
                reconstructed=transcript[:80],
                n=len(alt.words),
            )
        else:
            transcript = alt.transcript
        is_final = result.is_final
        await self._handle_transcript(transcript, is_final)

    async def _on_utt_end(self, result: Any) -> None:
        """Handle an utterance end event."""
        await self._handle_utterance_end()

    async def _handle_transcript(self, text: str, is_final: bool) -> None:
        """Invoke the transcript callback if registered."""
        if self._on_transcript:
            await self._on_transcript(text, is_final)

    async def _handle_utterance_end(self) -> None:
        """Invoke the utterance end callback if registered."""
        if self._on_utterance_end:
            await self._on_utterance_end()
