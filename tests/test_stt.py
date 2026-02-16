"""Tests for prot.stt — Deepgram Flux WebSocket STT client."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from prot.stt import STTClient


@pytest.mark.asyncio
class TestSTTClient:
    async def test_on_transcript_callback(self):
        transcripts = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test", on_transcript=on_transcript)
        await client._handle_transcript("테스트 문장", is_final=True)
        assert transcripts == [("테스트 문장", True)]

    async def test_on_utterance_end_callback(self):
        called = []

        async def on_end():
            called.append(True)

        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test", on_utterance_end=on_end)
        await client._handle_utterance_end()
        assert called == [True]

    def test_keyterms_configured(self):
        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(
                api_key="test",
                keyterms=["Axel", "NorthProt"],
            )
        assert client._keyterms == ["Axel", "NorthProt"]

    async def test_handle_transcript_skips_when_no_callback(self):
        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test")
        await client._handle_transcript("some text", is_final=False)

    async def test_handle_utterance_end_skips_when_no_callback(self):
        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test")
        await client._handle_utterance_end()

    def test_default_keyterms_empty(self):
        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test")
        assert client._keyterms == []

    async def test_on_message_extracts_transcript(self):
        transcripts = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test", on_transcript=on_transcript)

        mock_word = MagicMock()
        mock_word.punctuated_word = "안녕하세요"
        mock_word.word = "안녕하세요"
        mock_alt = MagicMock(transcript="안녕하세요", words=[mock_word])
        mock_result = MagicMock()
        mock_result.channel.alternatives = [mock_alt]
        mock_result.is_final = True

        await client._on_message(mock_result)
        assert transcripts == [("안녕하세요", True)]

    async def test_on_message_skips_empty_transcript(self):
        transcripts = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test", on_transcript=on_transcript)

        mock_result = MagicMock()
        mock_result.channel.alternatives = [MagicMock(transcript="")]
        mock_result.is_final = False

        await client._on_message(mock_result)
        assert transcripts == []

    async def test_on_utt_end_triggers_callback(self):
        called = []

        async def on_end():
            called.append(True)

        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test", on_utterance_end=on_end)

        await client._on_utt_end(MagicMock())
        assert called == [True]

    async def test_send_audio_calls_send_media(self):
        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test")
        mock_conn = MagicMock()
        client._connection = mock_conn
        await client.send_audio(b"\x00" * 512)
        mock_conn.send_media.assert_called_once_with(b"\x00" * 512)

    async def test_send_audio_noop_when_no_connection(self):
        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test")
        await client.send_audio(b"\x00" * 512)

    async def test_send_audio_disconnect_on_failure(self):
        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test")
        mock_conn = MagicMock()
        mock_conn.send_media = AsyncMock(side_effect=RuntimeError("boom"))
        client._connection = mock_conn
        client.disconnect = AsyncMock()

        await client.send_audio(b"\x00" * 512)
        client.disconnect.assert_awaited_once()

    async def test_disconnect_cancels_recv_task(self):
        with patch("prot.stt.AsyncDeepgramClient"):
            client = STTClient(api_key="test")
        mock_task = MagicMock()
        client._recv_task = mock_task
        client._connection_ctx = AsyncMock()
        await client.disconnect()
        mock_task.cancel.assert_called_once()
        assert client._recv_task is None
        assert client._connection is None
