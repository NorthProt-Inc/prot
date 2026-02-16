"""Pipeline orchestrator — wires all components together with state machine."""

from __future__ import annotations

import asyncio
import logging

from prot.config import settings
from prot.context import ContextManager
from prot.llm import LLMClient
from prot.persona import load_persona
from prot.playback import AudioPlayer
from prot.processing import chunk_sentences, sanitize_for_tts
from prot.state import State, StateMachine
from prot.stt import STTClient
from prot.tts import TTSClient
from prot.vad import VADProcessor

logger = logging.getLogger(__name__)


class Pipeline:
    """Core orchestrator wiring STT, LLM, TTS, VAD, and state machine."""

    def __init__(self) -> None:
        self._sm = StateMachine(
            vad_threshold_normal=settings.vad_threshold,
            vad_threshold_speaking=settings.vad_threshold_speaking,
        )
        self._vad = VADProcessor(threshold=settings.vad_threshold)
        self._stt = STTClient(
            on_transcript=self._on_transcript,
            on_utterance_end=self._handle_utterance_end,
        )
        self._llm = LLMClient()
        self._tts = TTSClient()
        self._player = AudioPlayer()
        self._ctx = ContextManager(persona_text=load_persona())

        # Optional components — initialized in startup()
        self._memory = None
        self._graphrag = None
        self._embedder = None
        self._pool = None

        self._current_transcript: str = ""
        self._active_timeout_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def state(self) -> StateMachine:
        """Expose state machine for external access (e.g., health endpoints)."""
        return self._sm

    async def startup(self) -> None:
        """Initialize optional async resources (DB, GraphRAG, embedder, memory)."""
        self._loop = asyncio.get_running_loop()
        try:
            from prot.db import init_pool
            self._pool = await init_pool()
        except Exception:
            logger.warning("DB pool not available — running without memory")
            return

        try:
            from prot.graphrag import GraphRAGStore
            from prot.embeddings import AsyncVoyageEmbedder
            from prot.memory import MemoryExtractor

            self._graphrag = GraphRAGStore(pool=self._pool)
            self._embedder = AsyncVoyageEmbedder()
            self._memory = MemoryExtractor(
                store=self._graphrag,
                embedder=self._embedder,
            )
        except Exception:
            logger.warning("Memory subsystem not available")

        # Pre-load RAG context
        try:
            if self._memory:
                rag_text = await self._memory.pre_load_context("general")
                self._ctx.update_rag_context(rag_text)
        except Exception:
            logger.debug("RAG pre-load failed", exc_info=True)

    def on_audio_chunk(self, data: bytes) -> None:
        """Sync callback from AudioManager (PyAudio thread) — schedules async processing."""
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._async_audio_chunk(data), self._loop)
        except RuntimeError:
            logger.debug("Event loop unavailable for audio chunk")

    async def _async_audio_chunk(self, data: bytes) -> None:
        """Process a single audio chunk: VAD check, forward to STT if needed."""
        try:
            # Update VAD threshold from state machine
            self._vad.threshold = self._sm.vad_threshold

            is_speech = self._vad.is_speech(data)

            if is_speech:
                state = self._sm.state
                if state in (State.IDLE, State.ACTIVE):
                    await self._handle_vad_speech()
                # Barge-in disabled — no echo cancellation, TTS output triggers false VAD
                # elif state == State.SPEAKING:
                #     self._sm.on_speech_detected()
                #     await self._handle_barge_in()

            # Forward audio to STT when listening
            if self._sm.state == State.LISTENING:
                await self._stt.send_audio(data)
        except Exception:
            logger.exception("Error in audio chunk processing")

    async def _handle_vad_speech(self) -> None:
        """VAD detected speech in IDLE/ACTIVE — transition and connect STT."""
        logger.info("VAD speech detected — connecting STT")
        self._sm.on_speech_detected()
        self._vad.reset()
        self._current_transcript = ""
        await self._stt.connect()

    async def _on_transcript(self, text: str, is_final: bool) -> None:
        """STT callback — store final transcript only."""
        if is_final:
            self._current_transcript += text
            logger.info("STT final: %s", text)

    async def _handle_utterance_end(self) -> None:
        """STT utterance end — transition to PROCESSING and run response."""
        if not self._current_transcript.strip():
            return

        logger.info("Utterance complete: %s", self._current_transcript.strip())
        self._sm.on_utterance_complete()
        await self._stt.disconnect()

        self._ctx.add_message("user", self._current_transcript)
        await self._process_response()

    async def _process_response(self) -> None:
        """Stream LLM -> TTS -> playback with sentence chunking."""
        system_blocks = self._ctx.build_system_blocks()
        tools = self._ctx.build_tools()
        messages = self._ctx.get_messages()

        full_response = ""
        buffer = ""

        try:
            logger.info("LLM streaming response...")
            self._sm.on_tts_started()
            await self._player.start()

            async for chunk in self._llm.stream_response(
                system_blocks, tools, messages
            ):
                if self._sm.state == State.INTERRUPTED:
                    break
                full_response += chunk
                buffer += chunk

                sentences, buffer = chunk_sentences(buffer)
                for sentence in sentences:
                    clean = sanitize_for_tts(sentence)
                    if not clean:
                        continue
                    async for audio_data in self._tts.stream_audio(clean):
                        if self._sm.state == State.INTERRUPTED:
                            break
                        await self._player.play_chunk(audio_data)

            # Flush remaining buffer
            if buffer.strip() and self._sm.state != State.INTERRUPTED:
                clean = sanitize_for_tts(buffer)
                if clean:
                    async for audio_data in self._tts.stream_audio(clean):
                        if self._sm.state == State.INTERRUPTED:
                            break
                        await self._player.play_chunk(audio_data)

            if self._sm.state != State.INTERRUPTED:
                await self._player.finish()

            if self._sm.state == State.SPEAKING:
                self._sm.on_tts_complete()
                self._ctx.add_message("assistant", full_response)
                logger.info("Response complete (%d chars) — state=ACTIVE", len(full_response))
                self._start_active_timeout()
                self._extract_memories_bg()

        except Exception:
            logger.exception("Error in _process_response")
            try:
                await self._player.kill()
            except Exception:
                pass

    async def _handle_barge_in(self) -> None:
        """User interrupted during TTS — cancel everything, reconnect STT."""
        self._llm.cancel()
        self._tts.flush()
        await self._player.kill()
        self._sm.on_interrupt_handled()
        self._vad.reset()
        self._current_transcript = ""
        await self._stt.connect()

    def _start_active_timeout(self) -> None:
        """Start timer to return to IDLE after active_timeout seconds."""
        if self._active_timeout_task is not None:
            self._active_timeout_task.cancel()

        async def _timeout() -> None:
            await asyncio.sleep(settings.active_timeout)
            if self._sm.state == State.ACTIVE:
                self._sm.on_active_timeout()
                await self._stt.disconnect()
                self._vad.reset()

        self._active_timeout_task = asyncio.ensure_future(_timeout())

    def _extract_memories_bg(self) -> None:
        """Background task to extract and save memories — silently handles failures."""
        if not self._memory:
            return

        async def _extract() -> None:
            try:
                messages = self._ctx.get_messages()
                extraction = await self._memory.extract_from_conversation(messages)
                await self._memory.save_extraction(extraction)

                # Refresh RAG context
                if self._current_transcript:
                    rag = await self._memory.pre_load_context(
                        self._current_transcript
                    )
                    self._ctx.update_rag_context(rag)
            except Exception:
                logger.debug("Memory extraction failed", exc_info=True)

        asyncio.ensure_future(_extract())

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        if self._active_timeout_task is not None:
            self._active_timeout_task.cancel()
            self._active_timeout_task = None

        try:
            await self._stt.disconnect()
        except Exception:
            logger.debug("STT disconnect error", exc_info=True)

        try:
            await self._player.kill()
        except Exception:
            logger.debug("Player kill error", exc_info=True)

        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                logger.debug("DB pool close error", exc_info=True)
