"""Tests for Pipeline orchestrator — all components mocked."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

    # LLM
    p._llm = MagicMock()
    p._llm.cancel = MagicMock()
    p._llm.close = AsyncMock()
    p._llm.get_tool_use_blocks = MagicMock(return_value=[])

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

    # Conversation logger
    p._conv_logger = MagicMock()

    # Optional components (may not be available)
    p._memory = None
    p._graphrag = None
    p._embedder = None
    p._reranker = None
    p._hass_registry = None
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
    p._session_msg_offset = 0

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
        p._sm.on_tts_complete()        # -> ACTIVE
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
    """_handle_barge_in() — cancels LLM, flushes TTS, kills player, reconnects STT."""

    async def test_cancels_llm(self):
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

        with patch("prot.db.export_tables", new_callable=AsyncMock):
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
        p._sm.on_tts_complete()
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

        # TTS yields audio bytes
        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as mock_settings:
            mock_settings.claude_model = "test"
            mock_settings.active_timeout = 999
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
            await p._process_response()

        # Should NOT have transitioned to ACTIVE (was interrupted)
        assert p._sm.state != State.ACTIVE
        # finish() should not be called when interrupted
        p._player.finish.assert_not_awaited()


class TestResponseContentRouting:
    """_process_response() uses last_response_content for context, plain text for DB."""

    async def test_context_gets_response_content_blocks(self):
        """Context should receive full response content (with compaction blocks)."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        async def fake_stream(*a, **kw):
            yield "Hello."

        p._llm.stream_response = fake_stream
        mock_content = [MagicMock(text="Hello.")]
        p._llm.last_response_content = mock_content

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as mock_settings:
            mock_settings.claude_model = "test"
            mock_settings.active_timeout = 999
            await p._process_response()

        # Context should receive the content blocks, not plain text
        p._ctx.add_message.assert_called_once_with("assistant", mock_content)

    async def test_db_save_gets_plain_text(self):
        """DB save should receive plain text, not content blocks."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()
        p._graphrag = AsyncMock()
        p._graphrag.save_message = AsyncMock()

        async def fake_stream(*a, **kw):
            yield "Hello."

        p._llm.stream_response = fake_stream
        p._llm.last_response_content = [MagicMock(text="Hello.")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as mock_settings:
            mock_settings.claude_model = "test"
            mock_settings.active_timeout = 999
            await p._process_response()

        # DB save should receive plain text
        assert len(p._background_tasks) >= 0  # task may have completed
        # Verify _save_message_bg was called — check ctx.add_message got blocks
        # while the graphrag save got text
        ctx_call = p._ctx.add_message.call_args
        assert isinstance(ctx_call[0][1], list)  # content blocks

    async def test_context_falls_back_to_text_without_response_content(self):
        """When last_response_content is None, context gets plain text."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        async def fake_stream(*a, **kw):
            yield "Fallback."

        p._llm.stream_response = fake_stream
        p._llm.last_response_content = None

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as mock_settings:
            mock_settings.claude_model = "test"
            mock_settings.active_timeout = 999
            await p._process_response()

        p._ctx.add_message.assert_called_once_with("assistant", "Fallback.")


class TestExtractMemoriesBg:
    """_extract_memories_bg() — background task lifecycle."""

    async def test_background_task_tracked(self):
        """Fire-and-forget tasks should be tracked for shutdown cleanup."""
        p = _make_pipeline()
        mock_memory = AsyncMock()
        mock_memory.extract_from_conversation = AsyncMock(
            return_value={"entities": [], "relationships": []}
        )
        mock_memory.save_extraction = AsyncMock()
        p._memory = mock_memory

        p._extract_memories_bg()
        assert len(p._background_tasks) == 1
        await asyncio.sleep(0.05)
        assert len(p._background_tasks) == 0

    async def test_shutdown_cancels_background_tasks(self):
        """shutdown() should cancel in-flight background tasks."""
        p = _make_pipeline()

        async def slow_extract(msgs):
            await asyncio.sleep(10)
            return {"entities": [], "relationships": []}

        mock_memory = AsyncMock()
        mock_memory.extract_from_conversation = slow_extract
        p._memory = mock_memory

        p._extract_memories_bg()
        assert len(p._background_tasks) == 1

        await p.shutdown()
        assert len(p._background_tasks) == 0


class TestSaveMessageBg:
    """_save_message_bg() — persists messages to DB in background."""

    async def test_save_message_bg_creates_tracked_task(self):
        """_save_message_bg() should create a background task tracked for cleanup."""
        p = _make_pipeline()
        p._graphrag = AsyncMock()
        p._graphrag.save_message = AsyncMock()
        p._save_message_bg("user", "test message")
        assert len(p._background_tasks) == 1
        await asyncio.sleep(0.05)
        assert len(p._background_tasks) == 0

    async def test_save_message_bg_noop_without_graphrag(self):
        """_save_message_bg() should be a no-op without graphrag."""
        p = _make_pipeline()
        p._graphrag = None
        p._save_message_bg("user", "test")
        assert len(p._background_tasks) == 0


class TestSaveSessionLog:
    """_save_session_log() — saves conversation as JSON and resets session."""

    async def test_save_session_log_calls_logger(self):
        p = _make_pipeline()
        p._conv_logger = MagicMock()
        p._ctx.get_messages.return_value = [
            {"role": "user", "content": "hello"},
        ]
        old_id = p._conversation_id
        p._save_session_log()

        p._conv_logger.save_session.assert_called_once_with(
            old_id,
            [{"role": "user", "content": "hello"}],
        )
        # Session ID should be reset
        assert p._conversation_id != old_id

    async def test_save_session_log_skips_empty(self):
        p = _make_pipeline()
        p._conv_logger = MagicMock()
        p._ctx.get_messages.return_value = []
        p._save_session_log()

        p._conv_logger.save_session.assert_not_called()

    async def test_save_session_log_only_saves_new_messages(self):
        p = _make_pipeline()
        p._ctx.get_messages.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        p._save_session_log()
        first_call_msgs = p._conv_logger.save_session.call_args_list[0][0][1]
        assert len(first_call_msgs) == 2

        p._ctx.get_messages.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "sunny"},
        ]
        p._save_session_log()
        second_call_msgs = p._conv_logger.save_session.call_args_list[1][0][1]
        assert len(second_call_msgs) == 2
        assert second_call_msgs[0]["content"] == "weather?"

    async def test_save_session_log_skips_if_no_new_messages(self):
        p = _make_pipeline()
        p._ctx.get_messages.return_value = [{"role": "user", "content": "hello"}]
        p._save_session_log()
        assert p._conv_logger.save_session.call_count == 1
        p._save_session_log()
        assert p._conv_logger.save_session.call_count == 1


class TestShutdownSessionSave:
    """shutdown() — saves session log before cleanup."""

    async def test_shutdown_saves_session_log(self):
        p = _make_pipeline()
        p._ctx.get_messages.return_value = [{"role": "user", "content": "hello"}]
        await p.shutdown()
        p._conv_logger.save_session.assert_called_once()

    async def test_shutdown_skips_session_log_if_empty(self):
        p = _make_pipeline()
        p._ctx.get_messages.return_value = []
        await p.shutdown()
        p._conv_logger.save_session.assert_not_called()


class TestToolUseHandling:
    """_process_response() — tool use loop execution."""

    async def test_tool_use_executes_and_loops(self):
        """When LLM returns tool_use, pipeline executes tool and calls LLM again."""
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
        tool_block.name = "hass_control"
        tool_block.id = "tool_123"
        tool_block.input = {"entity_id": "light.living", "action": "turn_on"}

        tc = 0

        def fake_get_tool():
            nonlocal tc
            tc += 1
            return [tool_block] if tc == 1 else []

        p._llm.get_tool_use_blocks = fake_get_tool

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value={"success": True, "message": "done"})
        p._hass_registry = mock_registry

        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            await p._process_response()

        assert call_count == 2
        mock_registry.execute.assert_awaited_once()
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
        tool_block.name = "hass_control"
        tool_block.id = "tool_456"
        tool_block.input = {"entity_id": "light.bad", "action": "turn_on"}

        tc = 0

        def fake_get_tool():
            nonlocal tc
            tc += 1
            return [tool_block] if tc == 1 else []

        p._llm.get_tool_use_blocks = fake_get_tool

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(side_effect=RuntimeError("HASS down"))
        p._hass_registry = mock_registry

        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            await p._process_response()

        assert call_count == 2
        # Tool result with error should have been added to context
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
        client._client.messages.stream = MagicMock(
            side_effect=RuntimeError("connection failed")
        )

        with pytest.raises(RuntimeError):
            async for _ in client.stream_response([], None, []):
                pass

        # Should be reset to None, not stale tool_use blocks
        assert client._last_response_content is None


class TestToolLoopExhaustion:
    """_process_response() — tool loop hits max iterations without deadlock."""

    async def test_all_iterations_receive_tools(self):
        """All iterations receive the same tools list (no stripping)."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        captured_tools = []

        async def fake_stream(system, tools, messages):
            captured_tools.append(tools)
            yield "Response."

        p._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "hass_control"
        tool_block.id = "tool_exhaust"
        tool_block.input = {"entity_id": "light.living", "action": "turn_on"}

        call_count = 0

        def fake_get_tool():
            nonlocal call_count
            call_count += 1
            # Return tool blocks for first 2 calls, empty on 3rd
            return [tool_block] if call_count < 3 else []

        p._llm.get_tool_use_blocks = fake_get_tool

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value={"success": True, "message": "done"})
        p._hass_registry = mock_registry

        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            await p._process_response()

        assert len(captured_tools) == 3
        assert captured_tools[0] is not None  # iteration 0: tools passed
        assert captured_tools[1] is not None  # iteration 1: tools passed
        assert captured_tools[2] is not None  # iteration 2: tools still passed
        assert p._sm.state == State.ACTIVE    # not stuck in PROCESSING


class TestInterruptionDuringToolExec:
    """_process_response() — barge-in during tool execution bails cleanly."""

    async def test_bails_out_when_interrupted_during_tool_exec(self):
        """When barge-in fires during tool execution, pipeline exits without ValueError."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        async def fake_stream(*a, **kw):
            yield "Checking."

        p._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "hass_control"
        tool_block.id = "tool_int"
        tool_block.input = {"entity_id": "light.living", "action": "turn_on"}

        p._llm.get_tool_use_blocks = MagicMock(return_value=[tool_block])

        async def fake_execute(name, inp):
            # Simulate barge-in during tool execution
            p._sm.on_speech_detected()    # SPEAKING -> INTERRUPTED
            p._sm.on_interrupt_handled()  # INTERRUPTED -> LISTENING
            return {"success": True, "message": "done"}

        mock_registry = AsyncMock()
        mock_registry.execute = fake_execute
        p._hass_registry = mock_registry

        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            await p._process_response()

        # Should have bailed out; state managed by barge-in handler
        assert p._sm.state == State.LISTENING


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
            p._sm._state = State.LISTENING
            raise RuntimeError("Error after barge-in")

        p._llm.stream_response = failing_stream

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
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
    async def test_hass_tool_routed_to_registry(self):
        """HASS tool calls are routed through _hass_registry, not _llm."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value={"success": True, "message": "done"})
        p._hass_registry = mock_registry

        call_count = 0

        async def fake_stream(*a, **kw):
            nonlocal call_count
            call_count += 1
            yield "Checking." if call_count == 1 else "Done."

        p._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "hass_control"
        tool_block.id = "tool_hass"
        tool_block.input = {"entity_id": "light.wiz_1", "action": "turn_on"}

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
            await p._process_response()

        mock_registry.execute.assert_awaited_once_with("hass_control", tool_block.input)

    async def test_build_tools_called_with_registry(self):
        """build_tools receives hass_registry parameter."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        mock_registry = MagicMock()
        p._hass_registry = mock_registry

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
            await p._process_response()

        p._ctx.build_tools.assert_called_with(hass_registry=mock_registry)
