"""Pipeline orchestrator — wires all components together with state machine."""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from prot.config import settings
from prot.context import ContextManager
from prot.engine import ConversationEngine, ToolIterationMarker
from prot.llm import LLMClient
from prot.persona import load_persona
from prot.playback import AudioPlayer
from prot.processing import chunk_sentences, sanitize_for_tts
from prot.state import State, StateMachine
from prot.stt import STTClient
from prot.tts import TTSClient
from prot.vad import VADProcessor
from prot.logging import get_logger, start_turn, reset_turn, logged

logger = get_logger(__name__)


class Pipeline:
    """Core orchestrator wiring STT, LLM, TTS, VAD, and state machine."""

    def __init__(self) -> None:
        self._sm = StateMachine(
            vad_threshold_normal=settings.vad_threshold,
            vad_threshold_speaking=settings.vad_threshold_speaking,
        )
        self._vad = VADProcessor(
            threshold=settings.vad_threshold,
            prebuffer_chunks=settings.vad_prebuffer_chunks,
        )
        self._stt = STTClient(
            on_transcript=self._on_transcript,
            on_utterance_end=self._handle_utterance_end,
        )
        self._llm = LLMClient()
        self._tts = TTSClient()
        self._player = AudioPlayer()
        self._ctx = ContextManager(persona_text=load_persona())
        self._engine = ConversationEngine(ctx=self._ctx, llm=self._llm)
        # Optional components — initialized in startup()
        self._memory = None
        self._graphrag = None
        self._embedder = None
        self._reranker = None
        self._hass_agent = None
        self._pool = None

        self._conversation_id = uuid4()
        self._current_transcript: str = ""
        self._pending_audio: list[bytes] = []
        self._stt_connected: bool = False
        self._active_timeout_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._barge_in_count: int = 0
        self._barge_in_frames: int = 6  # ~192ms sustained speech to trigger barge-in
        self._speaking_since: float = 0.0  # monotonic time when SPEAKING entered
        self._barge_in_grace: float = 1.5  # seconds to ignore VAD after SPEAKING starts
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def state(self) -> StateMachine:
        """Expose state machine for external access (e.g., health endpoints)."""
        return self._sm

    @property
    def current_state(self) -> str:
        return self._sm.state.value

    @logged(slow_ms=5000)
    async def startup(self) -> None:
        """Initialize optional async resources (HASS, DB, GraphRAG, embedder, memory)."""
        self._loop = asyncio.get_running_loop()

        try:
            from prot.hass import HassAgent
            if settings.hass_token:
                self._hass_agent = HassAgent(settings.hass_url, settings.hass_token, settings.hass_agent_id)
                logger.info("HASS agent ready")
        except Exception:
            logger.warning("HASS agent not available")

        try:
            from prot.db import init_pool
            self._pool = await init_pool()
        except Exception:
            logger.warning("DB pool not available — running without memory")
            return

        try:
            from prot.graphrag import MemoryStore
            from prot.embeddings import AsyncVoyageEmbedder
            from prot.memory import MemoryExtractor
            from prot.reranker import VoyageReranker

            self._graphrag = MemoryStore(pool=self._pool)
            self._embedder = AsyncVoyageEmbedder()
            self._reranker = VoyageReranker()
            self._memory = MemoryExtractor(
                store=self._graphrag,
                embedder=self._embedder,
                reranker=self._reranker,
            )
        except Exception:
            logger.warning("Memory subsystem not available")

        # Wire optional resources into the engine
        self._engine._hass_agent = self._hass_agent
        self._engine._memory = self._memory
        self._engine._graphrag = self._graphrag

        # Pre-warm TTS connection pool
        try:
            await self._tts.warm()
        except Exception:
            logger.debug("TTS warm failed", exc_info=True)

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
                    self._barge_in_count = 0
                    await self._handle_vad_speech()
                elif state == State.SPEAKING:
                    if settings.barge_in_enabled:
                        elapsed = time.monotonic() - self._speaking_since
                        if elapsed < self._barge_in_grace:
                            pass  # grace period — ignore VAD right after SPEAKING starts
                        else:
                            self._barge_in_count += 1
                            if self._barge_in_count >= self._barge_in_frames:
                                logger.info("Barge-in", frames=self._barge_in_count)
                                self._barge_in_count = 0
                                self._sm.on_speech_detected()
                                await self._handle_barge_in()
            else:
                self._barge_in_count = 0

            # Forward audio to STT when listening
            if self._sm.state == State.LISTENING:
                if self._stt_connected:
                    await self._stt.send_audio(data)
                else:
                    self._pending_audio.append(data)
        except Exception:
            logger.exception("Error in audio chunk processing")


    async def _reconnect_stt(self) -> bool:
        """Drain prebuffer, reset VAD, connect STT, and flush pending audio.

        Returns True on success, False if STT connection failed.
        On failure the caller is responsible for state transitions;
        this helper only cleans up pending audio and resets the turn.
        """
        self._pending_audio = self._vad.drain_prebuffer()
        self._vad.reset()

        await self._stt.connect()
        if not self._stt.is_connected:
            self._pending_audio.clear()
            reset_turn()
            return False

        for chunk in self._pending_audio:
            await self._stt.send_audio(chunk)
        self._pending_audio.clear()
        self._stt_connected = True
        return True

    async def _handle_vad_speech(self) -> None:
        """VAD detected speech in IDLE/ACTIVE — transition and connect STT."""
        start_turn()
        logger.info("VAD speech", state=self._sm.state.value)
        self._sm.on_speech_detected()
        self._current_transcript = ""
        self._stt_connected = False

        if not await self._reconnect_stt():
            logger.warning("STT connect failed, falling back to IDLE")
            self._sm.force_recovery(State.IDLE)
            return

    async def _on_transcript(self, text: str, is_final: bool) -> None:
        """STT callback — store final transcript only."""
        if is_final:
            if self._current_transcript:
                self._current_transcript += " "
            self._current_transcript += text
            logger.info("STT final", text=text[:50])

    @logged(slow_ms=500)
    async def _handle_utterance_end(self) -> None:
        """STT utterance end — transition to PROCESSING and run response."""
        if not self._current_transcript.strip():
            return

        logger.info("Utterance done", len=len(self._current_transcript.strip()))
        self._sm.on_utterance_complete()
        self._stt_connected = False
        self._pending_audio.clear()
        await self._stt.disconnect()

        self._engine.add_user_message(self._current_transcript)
        await self._process_response()

    @logged(slow_ms=2000)
    async def _process_response(self) -> None:
        """Voice: consume engine generator with concurrent TTS pipeline."""
        text_q: asyncio.Queue[str | None] = asyncio.Queue()
        audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
        tts_started = False

        sample_rate = int(settings.elevenlabs_output_format.split("_")[1])
        silence_ms = settings.tts_sentence_silence_ms
        silence = (
            b"\x00" * (int(sample_rate * silence_ms / 1000) * 2)
            if silence_ms > 0
            else b""
        )

        async def tts_producer() -> None:
            """text_q -> TTS -> audio_q (runs concurrently with engine)."""
            buffer = ""
            while True:
                chunk = await text_q.get()
                if chunk is None:
                    break
                if self._sm.state == State.INTERRUPTED:
                    break
                buffer += chunk
                sentences, buffer = chunk_sentences(buffer)
                for sentence in sentences:
                    clean = sanitize_for_tts(sentence)
                    if not clean:
                        continue
                    async for audio in self._tts.stream_audio(clean):
                        if self._sm.state == State.INTERRUPTED:
                            await audio_q.put(None)
                            return
                        await audio_q.put(audio)
                    if silence:
                        await audio_q.put(silence)
            # Flush remaining buffer
            if buffer.strip():
                clean = sanitize_for_tts(buffer)
                if clean:
                    async for audio in self._tts.stream_audio(clean):
                        if self._sm.state == State.INTERRUPTED:
                            break
                        await audio_q.put(audio)
            await audio_q.put(None)

        async def audio_consumer() -> None:
            """audio_q -> player."""
            first_chunk = True
            while True:
                data = await audio_q.get()
                if data is None:
                    break
                if self._sm.state == State.INTERRUPTED:
                    break
                if first_chunk:
                    self._speaking_since = time.monotonic()
                    first_chunk = False
                await self._player.play_chunk(data)

        tts_task = asyncio.create_task(tts_producer())
        consumer_task = asyncio.create_task(audio_consumer())

        try:
            async for item in self._engine.respond():
                if isinstance(item, ToolIterationMarker):
                    # Flush TTS pipeline, transition state
                    await text_q.put(None)
                    await tts_task
                    await audio_q.put(None)
                    await consumer_task
                    if tts_started and self._sm.state == State.SPEAKING:
                        self._sm.on_tool_iteration()
                    tts_started = False
                    # Restart TTS pipeline for next iteration
                    text_q = asyncio.Queue()
                    audio_q = asyncio.Queue(maxsize=32)
                    tts_task = asyncio.create_task(tts_producer())
                    consumer_task = asyncio.create_task(audio_consumer())
                    continue

                # Text chunk — check for interruption
                if self._sm.state == State.INTERRUPTED:
                    self._engine.cancel()
                    break

                if not tts_started:
                    self._sm.on_tts_started()
                    await self._player.start()
                    tts_started = True

                await text_q.put(item)

            # Signal end of text
            await text_q.put(None)
            await tts_task
            await consumer_task

            result = self._engine.last_result
            if result and not result.interrupted and tts_started:
                await self._player.finish()
                if self._sm.try_on_tts_complete():
                    reset_turn()
                    self._start_active_timeout()
                else:
                    reset_turn()
            else:
                reset_turn()

        except Exception:
            logger.exception("_process_response failed")
            try:
                await self._player.kill()
            except Exception:
                pass
            await text_q.put(None)
            await audio_q.put(None)
            try:
                await asyncio.wait_for(
                    asyncio.gather(tts_task, consumer_task, return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                tts_task.cancel()
                consumer_task.cancel()
            if self._sm.state in (State.PROCESSING, State.SPEAKING):
                self._sm.force_recovery(State.ACTIVE)
            reset_turn()

    @logged(slow_ms=200)
    async def _handle_barge_in(self) -> None:
        """User interrupted during TTS — cancel everything, reconnect STT."""
        logger.info("Interrupting", state=self._sm.state.value)
        self._engine.cancel()
        self._tts.flush()
        await self._player.kill()
        self._sm.on_interrupt_handled()
        self._stt_connected = False
        start_turn()
        self._current_transcript = ""

        if not await self._reconnect_stt():
            logger.warning("STT reconnect failed after barge-in, falling back to IDLE")
            self._sm.force_recovery(State.IDLE)
            return

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

        self._active_timeout_task = asyncio.create_task(_timeout())

    def _bg(self, coro) -> asyncio.Task:
        """Fire-and-forget a coroutine as a tracked background task."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def diagnostics(self) -> dict:
        """Return runtime diagnostics for monitoring."""
        diag = {
            "state": self._sm.state.value,
            "background_tasks": len(self._background_tasks),
            "active_timeout": self._active_timeout_task is not None,
            "asyncio_tasks": len(asyncio.all_tasks()),
        }
        if self._pool is not None:
            diag["db_pool_size"] = self._pool.get_size()
            diag["db_pool_free"] = self._pool.get_idle_size()
        return diag

    @logged(slow_ms=5000)
    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        self._stt_connected = False
        self._pending_audio.clear()

        if self._active_timeout_task is not None:
            self._active_timeout_task.cancel()
            try:
                await self._active_timeout_task
            except (asyncio.CancelledError, Exception):
                pass
            self._active_timeout_task = None

        # Cancel pipeline's own background tasks
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Best-effort shutdown summarization via engine
        await self._engine.shutdown_summarize()
        for task in self._engine._background_tasks:
            task.cancel()
        if self._engine._background_tasks:
            await asyncio.gather(*self._engine._background_tasks, return_exceptions=True)
        self._engine._background_tasks.clear()

        closeables = [
            self._llm.close,
            self._tts.close,
            self._stt.disconnect,
            self._player.kill,
        ]
        if self._memory is not None:
            closeables.append(self._memory.close)
        if self._embedder is not None:
            closeables.append(self._embedder.close)
        if self._hass_agent is not None:
            closeables.append(self._hass_agent.close)
        for close_fn in closeables:
            try:
                await close_fn()
            except Exception:
                logger.debug("Shutdown error", exc_info=True)

        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                logger.debug("DB pool close error", exc_info=True)
