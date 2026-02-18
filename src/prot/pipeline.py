"""Pipeline orchestrator — wires all components together with state machine."""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from prot.config import settings
from prot.context import ContextManager
from prot.conversation_log import ConversationLogger
from prot.llm import LLMClient
from prot.persona import load_persona
from prot.playback import AudioPlayer
from prot.processing import chunk_sentences
from prot.state import State, StateMachine
from prot.stt import STTClient
from prot.tts import TTSClient
from prot.vad import VADProcessor
from prot.log import get_logger, start_turn, reset_turn, logged

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
        self._conv_logger = ConversationLogger()

        # Optional components — initialized in startup()
        self._memory = None
        self._graphrag = None
        self._embedder = None
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
        self._session_msg_offset: int = 0

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
            from prot.community import CommunityDetector

            self._graphrag = GraphRAGStore(pool=self._pool)
            self._embedder = AsyncVoyageEmbedder()
            community_detector = CommunityDetector(
                store=self._graphrag,
                embedder=self._embedder,
            )
            self._memory = MemoryExtractor(
                store=self._graphrag,
                embedder=self._embedder,
                community_detector=community_detector,
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

    async def _handle_vad_speech(self) -> None:
        """VAD detected speech in IDLE/ACTIVE — transition and connect STT."""
        start_turn()
        logger.info("VAD speech", state=self._sm.state.value)
        self._sm.on_speech_detected()
        self._current_transcript = ""
        self._stt_connected = False

        # Drain ring buffer BEFORE reset (pre-trigger audio)
        self._pending_audio = self._vad.drain_prebuffer()
        self._vad.reset()

        await self._stt.connect()
        if not self._stt.is_connected:
            logger.warning("STT connect failed, falling back to IDLE")
            self._sm._state = State.IDLE
            self._pending_audio.clear()
            reset_turn()
            return

        # Flush prebuffer + audio that arrived during connect
        for chunk in self._pending_audio:
            await self._stt.send_audio(chunk)
        self._pending_audio.clear()
        self._stt_connected = True

    async def _on_transcript(self, text: str, is_final: bool) -> None:
        """STT callback — store final transcript only."""
        if is_final:
            if self._current_transcript:
                self._current_transcript += " "
            self._current_transcript += text
            logger.info("STT final", text=text[:50])

    async def _handle_utterance_end(self) -> None:
        """STT utterance end — transition to PROCESSING and run response."""
        if not self._current_transcript.strip():
            return

        logger.info("Utterance done", len=len(self._current_transcript.strip()))
        self._sm.on_utterance_complete()
        self._stt_connected = False
        self._pending_audio.clear()
        await self._stt.disconnect()

        self._ctx.add_message("user", self._current_transcript)
        self._save_message_bg("user", self._current_transcript)
        await self._process_response()

    _MAX_TOOL_ITERATIONS = 3

    @logged(slow_ms=2000)
    async def _process_response(self) -> None:
        """Stream LLM -> TTS -> playback with producer-consumer pipeline.

        Supports agentic tool loop: if the LLM returns tool_use blocks,
        tools are executed and results fed back for up to _MAX_TOOL_ITERATIONS.
        """
        system_blocks = self._ctx.build_system_blocks()
        tools = self._ctx.build_tools()

        try:
            for iteration in range(self._MAX_TOOL_ITERATIONS):
                messages = self._ctx.get_messages()

                # [FIX 1] Strip tools on final iteration to force text-only response
                iter_tools = tools if iteration < self._MAX_TOOL_ITERATIONS - 1 else None
                if iter_tools is None and tools:
                    logger.warning("Tool loop limit, forcing text-only", iteration=iteration)

                _response_parts: list[str] = []
                buffer = ""
                audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
                _last_pressure_log: float = 0.0

                async def _produce() -> None:
                    nonlocal buffer, _last_pressure_log
                    try:
                        async for chunk in self._llm.stream_response(
                            system_blocks, iter_tools, messages  # [FIX 1] iter_tools
                        ):
                            if self._sm.state == State.INTERRUPTED:
                                break
                            _response_parts.append(chunk)
                            buffer += chunk
                            sentences, buffer = chunk_sentences(buffer)
                            for sentence in sentences:
                                clean = sentence.strip()
                                if not clean:
                                    continue
                                async for audio in self._tts.stream_audio(clean):
                                    if self._sm.state == State.INTERRUPTED:
                                        return
                                    await audio_q.put(audio)
                                    qsz = audio_q.qsize()
                                    now = time.monotonic()
                                    if qsz >= 28 and now - _last_pressure_log >= 5.0:
                                        logger.warning("Queue pressure", qsize=qsz)
                                        _last_pressure_log = now
                        # Flush remaining
                        if buffer.strip() and self._sm.state != State.INTERRUPTED:
                            clean = buffer.strip()
                            if clean:
                                async for audio in self._tts.stream_audio(clean):
                                    if self._sm.state == State.INTERRUPTED:
                                        return
                                    await audio_q.put(audio)
                    finally:
                        await audio_q.put(None)  # sentinel

                async def _consume() -> None:
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

                logger.info("LLM streaming", model=settings.claude_model, iteration=iteration)

                # [FIX 2] Guard: bail if state changed (e.g. barge-in during previous tool exec)
                if self._sm.state != State.PROCESSING:
                    logger.info("State changed before TTS start",
                                iteration=iteration, state=self._sm.state.value)
                    reset_turn()
                    return

                self._sm.on_tts_started()  # PROCESSING -> SPEAKING
                await self._player.start()

                prod = asyncio.create_task(_produce())
                cons = asyncio.create_task(_consume())
                done, pending = await asyncio.wait(
                    [prod, cons], return_when=asyncio.FIRST_EXCEPTION,
                )
                for t in pending:
                    t.cancel()
                for t in done:
                    t.result()  # re-raise exceptions

                if self._sm.state != State.INTERRUPTED:
                    await self._player.finish()

                # Check for tool use
                tool_blocks = self._llm.get_tool_use_blocks()

                if not tool_blocks:
                    # Normal completion — no tools
                    if self._sm.try_on_tts_complete():
                        full_text = "".join(_response_parts)
                        response_content = self._llm.last_response_content
                        self._ctx.add_message("assistant", response_content or full_text)
                        self._save_message_bg("assistant", full_text)
                        logger.info("Response done", chars=len(full_text))
                        reset_turn()
                        self._start_active_timeout()
                        self._extract_memories_bg()
                    else:
                        # [FIX 3] Interrupted without tools — clean up turn timer
                        logger.info("Response interrupted", state=self._sm.state.value)
                        reset_turn()
                    return

                # Tool use detected — execute and loop
                logger.info("Tool use", count=len(tool_blocks), iteration=iteration)
                full_text = "".join(_response_parts)
                response_content = self._llm.last_response_content
                self._ctx.add_message("assistant", response_content or full_text)
                self._save_message_bg("assistant", full_text)

                tool_results = []
                for block in tool_blocks:
                    try:
                        result = await self._llm.execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                    except Exception as exc:
                        logger.warning("Tool failed", tool=block.name, error=str(exc))
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(exc),
                            "is_error": True,
                        })

                self._ctx.add_message("user", tool_results)

                # [FIX 4] Guard: bail if barge-in happened during tool execution
                if self._sm.state != State.SPEAKING:
                    logger.info("Interrupted during tool execution",
                                iteration=iteration, state=self._sm.state.value)
                    reset_turn()
                    return

                self._sm.on_tool_iteration()  # SPEAKING -> PROCESSING

            # [FIX 5] Defense-in-depth: for loop exhausted without returning
            # Should not reach here with tool stripping, but guard against it
            logger.error("Tool loop fell through without resolution")
            reset_turn()
            if self._sm.state in (State.PROCESSING, State.SPEAKING):
                self._sm._state = State.ACTIVE
                self._start_active_timeout()

        except Exception:
            logger.exception("Error in _process_response")
            try:
                await self._player.kill()
            except Exception:
                pass
            # [FIX 6] State-aware recovery — only force ACTIVE from stuck states
            reset_turn()
            if self._sm.state in (State.PROCESSING, State.SPEAKING):
                self._sm._state = State.ACTIVE
                self._start_active_timeout()

    async def _handle_barge_in(self) -> None:
        """User interrupted during TTS — cancel everything, reconnect STT."""
        logger.info("Interrupting", state=self._sm.state.value)
        self._llm.cancel()
        self._tts.flush()
        await self._player.kill()
        self._sm.on_interrupt_handled()
        self._stt_connected = False
        start_turn()
        self._current_transcript = ""

        self._pending_audio = self._vad.drain_prebuffer()
        self._vad.reset()

        await self._stt.connect()
        if not self._stt.is_connected:
            logger.warning("STT reconnect failed after barge-in, falling back to IDLE")
            self._sm._state = State.IDLE
            self._pending_audio.clear()
            reset_turn()
            return

        for chunk in self._pending_audio:
            await self._stt.send_audio(chunk)
        self._pending_audio.clear()
        self._stt_connected = True

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
                self._save_session_log()

        self._active_timeout_task = asyncio.create_task(_timeout())

    def _extract_memories_bg(self) -> None:
        """Background task to extract and save memories — tracked for shutdown cleanup."""
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
                logger.warning("Memory extraction failed", exc_info=True)

        task = asyncio.create_task(_extract())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _save_message_bg(self, role: str, content: str) -> None:
        """Persist a conversation message to DB in the background."""
        if not self._graphrag:
            return

        async def _save() -> None:
            try:
                await self._graphrag.save_message(
                    self._conversation_id, role, content
                )
            except Exception:
                logger.debug("Message save failed", exc_info=True)

        task = asyncio.create_task(_save())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _save_session_log(self) -> None:
        """Save new conversation messages as JSONL and reset session."""
        messages = self._ctx.get_messages()
        new_messages = messages[self._session_msg_offset:]
        if new_messages:
            self._conv_logger.save_session(self._conversation_id, new_messages)
        self._session_msg_offset = len(messages)
        self._conversation_id = uuid4()

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

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        self._stt_connected = False
        self._pending_audio.clear()
        self._save_session_log()

        if self._active_timeout_task is not None:
            self._active_timeout_task.cancel()
            try:
                await self._active_timeout_task
            except (asyncio.CancelledError, Exception):
                pass
            self._active_timeout_task = None

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
        for close_fn in closeables:
            try:
                await close_fn()
            except Exception:
                logger.debug("Shutdown error", exc_info=True)

        # Cancel background tasks before closing pool
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Export DB tables to CSV
        if self._pool is not None:
            try:
                from prot.db import export_tables
                await export_tables(self._pool)
            except Exception:
                logger.debug("DB export error", exc_info=True)

        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                logger.debug("DB pool close error", exc_info=True)
