"""Tests for Pipeline orchestrator — all components mocked."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prot.engine import ConversationEngine
from prot.state import State


def _make_pipeline():
    """Create a Pipeline instance with all components mocked, bypassing __init__."""
    from prot.pipeline import Pipeline

    p = Pipeline.__new__(Pipeline)

    # State machine — use the real one for state transition correctness
    from prot.state import StateMachine

    p._sm = StateMachine(vad_threshold_normal=0.5, vad_threshold_speaking=0.8)

    # VAD
    p._vad = MagicMock()
    p._vad.threshold = 0.5
    p._vad.is_speech = MagicMock(return_value=False)
    p._vad.reset = MagicMock()
    p._vad.drain_prebuffer = MagicMock(return_value=[])

    # STT
    p._stt = AsyncMock()
    p._stt.connect = AsyncMock()
    p._stt.send_audio = AsyncMock()
    p._stt.disconnect = AsyncMock()

    # LLM — mock with engine-required attributes
    p._llm = MagicMock()
    p._llm.cancel = MagicMock(side_effect=lambda: setattr(p._llm, "_cancelled", True))
    p._llm._cancelled = False
    p._llm.close = AsyncMock()
    p._llm.get_tool_use_blocks = MagicMock(return_value=[])
    p._llm.last_response_content = None
    p._llm.last_compaction_summary = None

    # TTS
    p._tts = MagicMock()
    p._tts.flush = MagicMock()

    # Player
    p._player = AsyncMock()
    p._player.start = AsyncMock()
    p._player.play_chunk = AsyncMock()
    p._player.finish = AsyncMock()
    p._player.kill = AsyncMock()

    # Context
    p._ctx = MagicMock()
    p._ctx.add_message = MagicMock()
    p._ctx.get_messages = MagicMock(return_value=[])
    p._ctx.get_recent_messages = MagicMock(return_value=[])
    p._ctx.build_system_blocks = MagicMock(return_value=[{"type": "text", "text": "test"}])
    p._ctx.build_tools = MagicMock(return_value=[])
    p._ctx.update_rag_context = MagicMock()

    # Engine — real engine wrapping mocked ctx and llm
    p._engine = ConversationEngine(ctx=p._ctx, llm=p._llm)

    # Optional components (may not be available)
    p._memory = None
    p._graphrag = None
    p._embedder = None
    p._reranker = None
    p._hass_agent = None
    p._pool = None

    # Internal state
    p._conversation_id = MagicMock()
    p._current_transcript = ""
    p._pending_audio = []
    p._stt_connected = False
    p._active_timeout_task: asyncio.Task | None = None
    p._loop = asyncio.get_running_loop()
    p._barge_in_count = 0
    p._barge_in_frames = 6
    p._speaking_since = 0.0
    p._barge_in_grace = 1.5
    p._background_tasks = set()

    return p


class TestHandleVadSpeech:
    """_handle_vad_speech() — VAD detected speech, transitions and connects STT."""

    async def test_transitions_from_idle_to_listening(self):
        p = _make_pipeline()
        assert p._sm.state == State.IDLE
        await p._handle_vad_speech()
        assert p._sm.state == State.LISTENING

    async def test_connects_stt(self):
        p = _make_pipeline()
        await p._handle_vad_speech()
        p._stt.connect.assert_awaited_once()

    async def test_resets_vad(self):
        p = _make_pipeline()
        await p._handle_vad_speech()
        p._vad.reset.assert_called_once()

    async def test_transitions_from_active_to_listening(self):
        p = _make_pipeline()
        # Move to ACTIVE state
        p._sm.on_speech_detected()   # IDLE -> LISTENING
        p._sm.on_utterance_complete()  # -> PROCESSING
        p._sm.on_tts_started()         # -> SPEAKING
        p._sm.try_on_tts_complete()        # -> ACTIVE
        assert p._sm.state == State.ACTIVE

        await p._handle_vad_speech()
        assert p._sm.state == State.LISTENING


class TestOnTranscript:
    """_on_transcript() — STT callback stores final transcript."""

    async def test_stores_final_transcript(self):
        p = _make_pipeline()
        await p._on_transcript("hello world", is_final=True)
        assert p._current_transcript == "hello world"

    async def test_ignores_interim_transcript(self):
        p = _make_pipeline()
        p._current_transcript = ""
        await p._on_transcript("partial", is_final=False)
        assert p._current_transcript == ""

    async def test_appends_multiple_finals(self):
        p = _make_pipeline()
        await p._on_transcript("hello", is_final=True)
        await p._on_transcript(" world", is_final=True)
        assert "hello" in p._current_transcript
        assert "world" in p._current_transcript


class TestHandleUtteranceEnd:
    """_handle_utterance_end() — STT utterance end triggers processing."""

    async def test_transitions_to_processing(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()  # IDLE -> LISTENING
        p._current_transcript = "test query"

        # Mock _process_response to avoid actual LLM call
        p._process_response = AsyncMock()
        await p._handle_utterance_end()

        assert p._sm.state == State.PROCESSING

    async def test_disconnects_stt(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._current_transcript = "test"
        p._process_response = AsyncMock()

        await p._handle_utterance_end()
        p._stt.disconnect.assert_awaited_once()

    async def test_calls_process_response(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._current_transcript = "test"
        p._process_response = AsyncMock()

        await p._handle_utterance_end()
        p._process_response.assert_awaited_once()

    async def test_skips_empty_transcript(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._current_transcript = ""
        p._process_response = AsyncMock()

        await p._handle_utterance_end()
        # Should not transition or call _process_response with empty transcript
        p._process_response.assert_not_awaited()

    async def test_delegates_user_message_to_engine(self):
        """_handle_utterance_end uses engine.add_user_message, not ctx directly."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._current_transcript = "test message"
        p._process_response = AsyncMock()

        await p._handle_utterance_end()
        # The engine should have added the user message to ctx
        p._ctx.add_message.assert_called_once_with("user", "test message")


class TestOnAudioChunk:
    """on_audio_chunk() — forwards audio to STT when LISTENING."""

    async def test_forwards_audio_when_listening(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()  # IDLE -> LISTENING
        p._stt_connected = True
        assert p._sm.state == State.LISTENING

        await p._async_audio_chunk(b"\x00" * 512)
        p._stt.send_audio.assert_awaited_once_with(b"\x00" * 512)

    async def test_runs_vad_check(self):
        p = _make_pipeline()
        await p._async_audio_chunk(b"\x00" * 512)
        p._vad.is_speech.assert_called_once()

    async def test_triggers_vad_speech_on_detection(self):
        p = _make_pipeline()
        p._vad.is_speech.return_value = True
        p._handle_vad_speech = AsyncMock()

        await p._async_audio_chunk(b"\x00" * 512)
        p._handle_vad_speech.assert_awaited_once()

    async def test_updates_vad_threshold_from_state(self):
        p = _make_pipeline()
        # Move to SPEAKING state (higher threshold)
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._sm.on_tts_started()
        assert p._sm.state == State.SPEAKING

        await p._async_audio_chunk(b"\x00" * 512)
        # VAD threshold should be set to the speaking threshold
        assert p._vad.threshold == p._sm.vad_threshold


class TestHandleBargeIn:
    """_handle_barge_in() — cancels engine, flushes TTS, kills player, reconnects STT."""

    async def test_cancels_engine(self):
        p = _make_pipeline()
        # Get to INTERRUPTED state
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._sm.on_tts_started()
        p._sm.on_speech_detected()  # -> INTERRUPTED
        assert p._sm.state == State.INTERRUPTED

        await p._handle_barge_in()
        p._llm.cancel.assert_called_once()

    async def test_flushes_tts(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._sm.on_tts_started()
        p._sm.on_speech_detected()

        await p._handle_barge_in()
        p._tts.flush.assert_called_once()

    async def test_kills_player(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._sm.on_tts_started()
        p._sm.on_speech_detected()

        await p._handle_barge_in()
        p._player.kill.assert_awaited_once()

    async def test_reconnects_stt(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._sm.on_tts_started()
        p._sm.on_speech_detected()

        await p._handle_barge_in()
        p._stt.connect.assert_awaited_once()

    async def test_transitions_to_listening(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._sm.on_tts_started()
        p._sm.on_speech_detected()

        await p._handle_barge_in()
        assert p._sm.state == State.LISTENING


class TestShutdown:
    """shutdown() — cleans up all resources."""

    async def test_disconnects_stt(self):
        p = _make_pipeline()
        await p.shutdown()
        p._stt.disconnect.assert_awaited_once()

    async def test_kills_player(self):
        p = _make_pipeline()
        await p.shutdown()
        p._player.kill.assert_awaited_once()

    async def test_cancels_active_timeout(self):
        p = _make_pipeline()
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        p._active_timeout_task = mock_task

        await p.shutdown()
        mock_task.cancel.assert_called_once()

    async def test_closes_db_pool(self):
        p = _make_pipeline()
        mock_pool = AsyncMock()
        p._pool = mock_pool

        await p.shutdown()
        mock_pool.close.assert_awaited_once()

    async def test_closes_tts_client(self):
        p = _make_pipeline()
        p._tts.close = AsyncMock()

        await p.shutdown()
        p._tts.close.assert_awaited_once()

    async def test_closes_memory_client(self):
        p = _make_pipeline()
        mock_memory = AsyncMock()
        mock_memory.close = AsyncMock()
        p._memory = mock_memory

        await p.shutdown()
        mock_memory.close.assert_awaited_once()

    async def test_closes_embedder_client(self):
        p = _make_pipeline()
        mock_embedder = AsyncMock()
        mock_embedder.close = AsyncMock()
        p._embedder = mock_embedder

        await p.shutdown()
        mock_embedder.close.assert_awaited_once()

    async def test_skips_memory_close_when_none(self):
        p = _make_pipeline()
        p._tts.close = AsyncMock()
        p._memory = None
        p._embedder = None

        await p.shutdown()  # should not raise


class TestActiveTimeout:
    """_start_active_timeout() — schedules 30s timer to return to IDLE."""

    async def test_cancels_previous_timeout(self):
        p = _make_pipeline()
        old_task = MagicMock()
        old_task.cancel = MagicMock()
        p._active_timeout_task = old_task

        p._start_active_timeout()
        old_task.cancel.assert_called_once()

    async def test_creates_new_task(self):
        p = _make_pipeline()
        p._start_active_timeout()
        assert p._active_timeout_task is not None

    async def test_timeout_transitions_to_idle(self):
        """After timeout fires, state goes ACTIVE -> IDLE."""
        p = _make_pipeline()
        # Move to ACTIVE
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._sm.on_tts_started()
        p._sm.try_on_tts_complete()
        assert p._sm.state == State.ACTIVE

        # Use a short timeout for testing — keep patch active while task runs
        with patch("prot.pipeline.settings") as mock_settings:
            mock_settings.active_timeout = 0  # immediate
            p._start_active_timeout()
            # Let the timeout task run within the patch context
            await asyncio.sleep(0.05)

        assert p._sm.state == State.IDLE


class TestProcessResponse:
    """_process_response() — producer-consumer pipeline tests."""

    async def test_process_response_plays_audio(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()     # IDLE -> LISTENING
        p._sm.on_utterance_complete()  # -> PROCESSING

        # LLM yields one chunk that forms a complete sentence
        async def fake_stream(*a, **kw):
            yield "Hello world."

        p._llm.stream_response = fake_stream
        p._llm.last_response_content = "Hello world."

        # TTS yields audio bytes
        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as mock_settings:
            mock_settings.claude_model = "test"
            mock_settings.active_timeout = 999
            mock_settings.tts_sentence_silence_ms = 0
            mock_settings.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        p._player.start.assert_awaited_once()
        p._player.play_chunk.assert_awaited()
        assert p._sm.state == State.ACTIVE

    async def test_process_response_interruption_stops_playback(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        # LLM yields chunks; mid-stream we set state to INTERRUPTED
        call_count = 0

        async def fake_stream(*a, **kw):
            nonlocal call_count
            for word in ["Hello ", "world. ", "More ", "text."]:
                call_count += 1
                if call_count == 2:
                    # Simulate barge-in: SPEAKING -> INTERRUPTED
                    p._sm.on_speech_detected()
                yield word

        p._llm.stream_response = fake_stream

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as mock_settings:
            mock_settings.claude_model = "test"
            mock_settings.active_timeout = 999
            mock_settings.tts_sentence_silence_ms = 0
            mock_settings.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        # Should NOT have transitioned to ACTIVE (was interrupted)
        assert p._sm.state != State.ACTIVE
        # finish() should not be called when interrupted
        p._player.finish.assert_not_awaited()


class TestSilencePadding:
    """tts_producer() — inter-sentence silence padding."""

    async def test_silence_between_sentences(self):
        """Two sentences should have a silence buffer between them."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        async def fake_stream(*a, **kw):
            yield "First sentence. Second sentence."

        p._llm.stream_response = fake_stream
        p._llm.last_response_content = "First sentence. Second sentence."

        tts_call_count = 0

        async def fake_tts(text):
            nonlocal tts_call_count
            tts_call_count += 1
            yield b"\xaa" * 100

        p._tts.stream_audio = fake_tts

        played_chunks: list[bytes] = []
        original_play = p._player.play_chunk

        async def capture_play(data):
            played_chunks.append(data)

        p._player.play_chunk = capture_play

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 200
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        assert tts_call_count == 2
        # Expected: audio1, silence, audio2, silence
        # (silence after every sentence — last one is harmless trailing gap)
        # Silence = 24000 * 200/1000 * 2 = 9600 bytes of zeros
        silence = b"\x00" * 9600
        assert len(played_chunks) == 4
        assert played_chunks[0] == b"\xaa" * 100
        assert played_chunks[1] == silence
        assert played_chunks[2] == b"\xaa" * 100
        assert played_chunks[3] == silence

    async def test_no_silence_when_disabled(self):
        """When tts_sentence_silence_ms=0, no silence is inserted."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        async def fake_stream(*a, **kw):
            yield "First sentence. Second sentence."

        p._llm.stream_response = fake_stream
        p._llm.last_response_content = "First sentence. Second sentence."

        async def fake_tts(text):
            yield b"\xaa" * 100

        p._tts.stream_audio = fake_tts

        played_chunks: list[bytes] = []

        async def capture_play(data):
            played_chunks.append(data)

        p._player.play_chunk = capture_play

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        # Only audio chunks, no silence
        assert all(chunk == b"\xaa" * 100 for chunk in played_chunks)
        assert len(played_chunks) == 2


class TestToolUseHandling:
    """_process_response() with engine — tool use loop execution."""

    async def test_tool_use_executes_and_loops(self):
        """When LLM returns tool_use, engine executes tool and calls LLM again."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        call_count = 0

        async def fake_stream(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Let me check."
            else:
                yield "The light is on."

        p._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "hass_request"
        tool_block.id = "tool_123"
        tool_block.input = {"command": "거실 조명 켜줘"}

        tc = 0

        def fake_get_tool():
            nonlocal tc
            tc += 1
            return [tool_block] if tc == 1 else []

        p._llm.get_tool_use_blocks = fake_get_tool

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value="done")
        p._hass_agent = mock_agent
        p._engine._hass_agent = mock_agent

        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        assert call_count == 2
        mock_agent.request.assert_awaited_once()
        assert p._sm.state == State.ACTIVE

    async def test_tool_use_error_is_reported(self):
        """When tool execution fails, error is sent back as tool_result."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        call_count = 0

        async def fake_stream(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Checking."
            else:
                yield "Sorry, error."

        p._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "hass_request"
        tool_block.id = "tool_456"
        tool_block.input = {"command": "잘못된 명령"}

        tc = 0

        def fake_get_tool():
            nonlocal tc
            tc += 1
            return [tool_block] if tc == 1 else []

        p._llm.get_tool_use_blocks = fake_get_tool

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=RuntimeError("HASS down"))
        p._hass_agent = mock_agent
        p._engine._hass_agent = mock_agent

        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        assert call_count == 2
        # Tool result with error should have been added to context via engine
        user_calls = [c for c in p._ctx.add_message.call_args_list if c[0][0] == "user"]
        assert len(user_calls) >= 1
        tool_result = user_calls[0][0][1]
        assert any("is_error" in r for r in tool_result)


class TestStreamResponseResetContent:
    """stream_response resets _last_response_content before streaming."""

    async def test_last_response_content_reset_before_stream(self):
        """_last_response_content is None if stream fails before completion."""
        from prot.llm import LLMClient

        client = LLMClient.__new__(LLMClient)
        client._cancelled = False
        client._active_stream = None
        # Simulate stale data from previous iteration
        client._last_response_content = [MagicMock(type="tool_use")]

        # Mock the Anthropic client to raise before streaming
        client._client = MagicMock()
        client._client.beta.messages.stream = MagicMock(
            side_effect=RuntimeError("connection failed")
        )

        with pytest.raises(RuntimeError):
            async for _ in client.stream_response([], None, []):
                pass

        # Should be reset to None, not stale tool_use blocks
        assert client._last_response_content is None


class TestExceptionRecovery:
    """_process_response() — exception recovers state appropriately."""

    async def test_exception_from_speaking_recovers_to_active(self):
        """When streaming raises from SPEAKING, state is recovered to ACTIVE."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        async def failing_stream(*a, **kw):
            yield "Hello"
            raise RuntimeError("API error")

        p._llm.stream_response = failing_stream

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        assert p._sm.state == State.ACTIVE

    async def test_exception_after_barge_in_preserves_listening(self):
        """When exception fires after barge-in moved state to LISTENING, don't force ACTIVE."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        async def failing_stream(*a, **kw):
            yield "Hello"
            # Simulate: barge-in already happened, state is LISTENING
            p._sm.force_recovery(State.LISTENING)
            raise RuntimeError("Error after barge-in")

        p._llm.stream_response = failing_stream

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        # State should remain LISTENING, not forced to ACTIVE
        assert p._sm.state == State.LISTENING


class TestSTTConnectFallback:
    """Pipeline falls back to IDLE when STT connect fails."""

    async def test_vad_speech_stt_connect_failure_falls_back_to_idle(self):
        p = _make_pipeline()
        p._stt.is_connected = False
        assert p._sm.state == State.IDLE

        await p._handle_vad_speech()
        assert p._sm.state == State.IDLE

    async def test_barge_in_stt_reconnect_failure_falls_back_to_idle(self):
        p = _make_pipeline()
        # Get to INTERRUPTED state
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._sm.on_tts_started()
        p._sm.on_speech_detected()  # -> INTERRUPTED
        assert p._sm.state == State.INTERRUPTED

        p._stt.is_connected = False
        await p._handle_barge_in()
        assert p._sm.state == State.IDLE


class TestPreBuffer:
    """Audio pre-buffering: ring buffer drain + pending queue during STT connect."""

    async def test_prebuffer_flushed_to_stt_on_speech(self):
        p = _make_pipeline()
        p._vad.drain_prebuffer = MagicMock(return_value=[b"pre1", b"pre2"])
        p._stt.is_connected = True
        await p._handle_vad_speech()
        calls = p._stt.send_audio.await_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == b"pre1"
        assert calls[1].args[0] == b"pre2"

    async def test_pending_audio_queued_during_connect(self):
        p = _make_pipeline()
        p._vad.drain_prebuffer = MagicMock(return_value=[])

        async def slow_connect():
            p._pending_audio.append(b"during1")

        p._stt.connect = AsyncMock(side_effect=slow_connect)
        p._stt.is_connected = True
        await p._handle_vad_speech()
        calls = p._stt.send_audio.await_args_list
        assert len(calls) == 1
        assert calls[0].args[0] == b"during1"

    async def test_stt_connected_false_routes_to_pending(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._stt_connected = False
        await p._async_audio_chunk(b"\x00" * 512)
        assert b"\x00" * 512 in p._pending_audio
        p._stt.send_audio.assert_not_awaited()

    async def test_stt_connected_true_routes_to_stt(self):
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._stt_connected = True
        await p._async_audio_chunk(b"\x00" * 512)
        p._stt.send_audio.assert_awaited_once()
        assert len(p._pending_audio) == 0

    async def test_pending_cleared_after_flush(self):
        p = _make_pipeline()
        p._vad.drain_prebuffer = MagicMock(return_value=[b"x"])
        p._stt.is_connected = True
        await p._handle_vad_speech()
        assert len(p._pending_audio) == 0
        assert p._stt_connected is True

    async def test_connect_failure_clears_pending(self):
        p = _make_pipeline()
        p._vad.drain_prebuffer = MagicMock(return_value=[b"x", b"y"])
        p._stt.is_connected = False
        await p._handle_vad_speech()
        assert len(p._pending_audio) == 0
        assert p._stt_connected is False


class TestPipelineHassRouting:
    async def test_hass_tool_routed_to_agent(self):
        """hass_request tool calls are routed through engine's _hass_agent."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value="조명을 켰습니다")
        p._hass_agent = mock_agent
        p._engine._hass_agent = mock_agent

        call_count = 0

        async def fake_stream(*a, **kw):
            nonlocal call_count
            call_count += 1
            yield "Checking." if call_count == 1 else "Done."

        p._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "hass_request"
        tool_block.id = "tool_hass"
        tool_block.input = {"command": "거실 조명 켜줘"}

        tc = 0

        def fake_get_tool():
            nonlocal tc
            tc += 1
            return [tool_block] if tc == 1 else []

        p._llm.get_tool_use_blocks = fake_get_tool
        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        mock_agent.request.assert_awaited_once_with("거실 조명 켜줘")

    async def test_build_tools_called_with_agent(self):
        """build_tools receives hass_agent parameter via engine."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        mock_agent = MagicMock()
        p._hass_agent = mock_agent
        p._engine._hass_agent = mock_agent

        async def fake_stream(*a, **kw):
            yield "Hello."

        p._llm.stream_response = fake_stream
        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        p._ctx.build_tools.assert_called_with(hass_agent=mock_agent)


class TestShutdownFinalExtraction:
    """shutdown() — delegates shutdown summarization to engine."""

    async def test_shutdown_runs_summarization_via_engine(self):
        p = _make_pipeline()
        mock_memory = AsyncMock()
        mock_memory.generate_shutdown_summary = AsyncMock(return_value="Summary text")
        mock_memory.extract_from_summary = AsyncMock(
            return_value={"semantic": [], "episodic": None, "emotional": [], "procedural": []}
        )
        mock_memory.save_extraction = AsyncMock()
        mock_memory.close = AsyncMock()
        p._memory = mock_memory
        p._engine._memory = mock_memory
        p._ctx.get_messages.return_value = [{"role": "user", "content": "hello"}]

        await p.shutdown()

        mock_memory.generate_shutdown_summary.assert_awaited_once()
        mock_memory.extract_from_summary.assert_awaited_once_with("Summary text")
        mock_memory.save_extraction.assert_awaited_once()

    async def test_shutdown_skips_without_memory(self):
        p = _make_pipeline()
        p._memory = None
        p._engine._memory = None
        p._tts.close = AsyncMock()

        await p.shutdown()  # should not raise


class TestPipelineUsesEngine:
    """Pipeline creates and uses a ConversationEngine."""

    async def test_has_engine_attribute(self):
        p = _make_pipeline()
        assert hasattr(p, "_engine")
        assert isinstance(p._engine, ConversationEngine)

    async def test_engine_shares_llm_and_ctx(self):
        p = _make_pipeline()
        assert p._engine._llm is p._llm
        assert p._engine._ctx is p._ctx
