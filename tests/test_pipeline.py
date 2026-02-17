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

    # STT
    p._stt = AsyncMock()
    p._stt.connect = AsyncMock()
    p._stt.send_audio = AsyncMock()
    p._stt.disconnect = AsyncMock()

    # LLM
    p._llm = MagicMock()
    p._llm.cancel = MagicMock()

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
    p._ctx.build_system_blocks = MagicMock(return_value=[{"type": "text", "text": "test"}])
    p._ctx.build_tools = MagicMock(return_value=[])
    p._ctx.update_rag_context = MagicMock()

    # Optional components (may not be available)
    p._memory = None
    p._graphrag = None
    p._embedder = None
    p._pool = None

    # Internal state
    p._current_transcript = ""
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

        await p.shutdown()
        mock_pool.close.assert_awaited_once()


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
