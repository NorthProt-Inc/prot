"""Deepgram Flux WebSocket streaming STT client."""

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

from prot.config import settings


class STTClient:
    """Deepgram Flux WebSocket streaming client."""

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
        self._connection_ctx = self._client.listen.v1.connect(
            model=settings.deepgram_model,
            language=settings.deepgram_language,
            smart_format="true",
            interim_results="true",
            utterance_end_ms="1000",
            endpointing=str(settings.deepgram_endpointing),
            keyterm=self._keyterms if self._keyterms else None,
        )
        self._connection = await self._connection_ctx.__aenter__()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def send_audio(self, data: bytes) -> None:
        """Send PCM audio chunk to Deepgram."""
        if self._connection:
            self._connection.send_media(data)

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        if self._connection_ctx:
            await self._connection_ctx.__aexit__(None, None, None)
            self._connection = None
            self._connection_ctx = None

    async def _recv_loop(self) -> None:
        """Receive and dispatch messages from Deepgram."""
        while self._connection:
            try:
                result = await self._connection.recv()
            except Exception:
                break
            if isinstance(result, ListenV1ResultsEvent):
                await self._on_message(result)
            elif isinstance(result, ListenV1UtteranceEndEvent):
                await self._on_utt_end(result)

    async def _on_message(self, result: ListenV1ResultsEvent) -> None:
        """Handle a transcript result event."""
        transcript = result.channel.alternatives[0].transcript
        if not transcript:
            return
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
