"""Tests for prot.stt — ElevenLabs Scribe v2 Realtime WebSocket STT client."""

import asyncio
import base64
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prot.stt import STTClient

_SESSION_MSG = json.dumps({
    "message_type": "session_started",
    "session_id": "test-session",
    "config": {},
})


def _make_ws_mock(recv_messages=None):
    """Create a mock WebSocket with configurable recv sequence."""
    ws = AsyncMock()
    ws.close = AsyncMock()
    side = [_SESSION_MSG]
    if recv_messages:
        side.extend(recv_messages)
    side.append(asyncio.CancelledError())
    ws.recv = AsyncMock(side_effect=side)
    ws.send = AsyncMock()
    return ws


@pytest.mark.asyncio
class TestSTTClient:
    async def test_send_audio_encodes_base64_json(self):
        ws = _make_ws_mock()
        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            client = STTClient(api_key="test")
            await client.connect()
            await client.send_audio(b"\x00" * 512)
            await client.disconnect()

        call_args = ws.send.call_args[0][0]
        msg = json.loads(call_args)
        assert msg["message_type"] == "input_audio_chunk"
        assert msg["sample_rate"] == 16000
        assert msg["commit"] is False
        raw = base64.b64decode(msg["audio_base_64"])
        assert raw == b"\x00" * 512

    async def test_send_audio_noop_when_no_connection(self):
        client = STTClient(api_key="test")
        await client.send_audio(b"\x00" * 512)  # should not raise

    async def test_partial_transcript_callback(self):
        transcripts = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        ws = _make_ws_mock([
            json.dumps({"message_type": "partial_transcript", "text": "안녕하세요"}),
        ])

        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            client = STTClient(api_key="test", on_transcript=on_transcript)
            await client.connect()
            await asyncio.sleep(0.05)
            await client.disconnect()

        assert ("안녕하세요", False) in transcripts

    async def test_committed_transcript_callback(self):
        transcripts = []
        utt_ends = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        async def on_utt_end():
            utt_ends.append(True)

        ws = _make_ws_mock([
            json.dumps({"message_type": "committed_transcript", "text": "테스트 문장입니다"}),
        ])

        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            client = STTClient(
                api_key="test",
                on_transcript=on_transcript,
                on_utterance_end=on_utt_end,
            )
            await client.connect()
            await asyncio.sleep(0.05)
            await client.disconnect()

        assert ("테스트 문장입니다", True) in transcripts
        assert utt_ends == [True]

    async def test_empty_transcript_skipped(self):
        transcripts = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        ws = _make_ws_mock([
            json.dumps({"message_type": "partial_transcript", "text": ""}),
            json.dumps({"message_type": "committed_transcript", "text": ""}),
        ])

        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            client = STTClient(api_key="test", on_transcript=on_transcript)
            await client.connect()
            await asyncio.sleep(0.05)
            await client.disconnect()

        assert transcripts == []

    async def test_disconnect_cancels_recv_task(self):
        ws = _make_ws_mock()
        # Override recv to block forever after session_started
        future = asyncio.get_event_loop().create_future()
        ws.recv = AsyncMock(side_effect=[_SESSION_MSG, future])

        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            client = STTClient(api_key="test")
            await client.connect()
            assert client._recv_task is not None
            await client.disconnect()
            assert client._recv_task is None
            assert client._ws is None

    async def test_disconnect_noop_when_not_connected(self):
        client = STTClient(api_key="test")
        await client.disconnect()  # should not raise

    async def test_no_callbacks_no_error(self):
        ws = _make_ws_mock([
            json.dumps({"message_type": "committed_transcript", "text": "hello"}),
        ])

        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            client = STTClient(api_key="test")
            await client.connect()
            await asyncio.sleep(0.05)
            await client.disconnect()

    async def test_send_audio_disconnect_on_failure(self):
        ws = _make_ws_mock()
        ws.send = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            client = STTClient(api_key="test")
            await client.connect()
            client.disconnect = AsyncMock()
            await client.send_audio(b"\x00" * 512)
            client.disconnect.assert_awaited_once()

    async def test_connect_reuses_when_already_connected(self):
        ws = _make_ws_mock()
        ws.open = True
        mock_connect = AsyncMock(return_value=ws)
        with patch("prot.stt.websockets.connect", mock_connect):
            client = STTClient(api_key="test")
            await client.connect()
            await client.connect()  # should reuse, not reconnect
            assert mock_connect.call_count == 1
            await client.disconnect()

    async def test_connect_reconnects_when_ws_closed(self):
        ws1 = _make_ws_mock()
        ws1.open = False
        ws2 = _make_ws_mock()
        mock_connect = AsyncMock(side_effect=[ws1, ws2])
        with patch("prot.stt.websockets.connect", mock_connect):
            client = STTClient(api_key="test")
            await client.connect()
            # ws1.open is False, so reconnect should happen
            await client.connect()
            assert mock_connect.call_count == 2

    async def test_connect_failure_closes_ws_and_sets_none(self):
        ws = AsyncMock()
        ws.recv = AsyncMock(side_effect=RuntimeError("handshake failed"))
        ws.close = AsyncMock()
        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)), \
             patch("prot.stt.asyncio.sleep", new_callable=AsyncMock):
            client = STTClient(api_key="test")
            await client.connect()
            assert client._ws is None
            assert client._recv_task is None
            # close called once per retry attempt (4 total)
            assert ws.close.await_count == 4

    async def test_connect_ws_failure_sets_none(self):
        mock_connect = AsyncMock(side_effect=ConnectionRefusedError("refused"))
        with patch("prot.stt.websockets.connect", mock_connect):
            client = STTClient(api_key="test")
            await client.connect()
            assert client._ws is None
            assert client._recv_task is None

    async def test_committed_transcript_with_timestamps(self):
        transcripts = []
        utt_ends = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        async def on_utt_end():
            utt_ends.append(True)

        ws = _make_ws_mock([
            json.dumps({
                "message_type": "committed_transcript_with_timestamps",
                "text": "타임스탬프 테스트",
                "words": [{"text": "타임스탬프", "start": 0.0, "end": 0.5}],
            }),
        ])

        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            client = STTClient(
                api_key="test",
                on_transcript=on_transcript,
                on_utterance_end=on_utt_end,
            )
            await client.connect()
            await asyncio.sleep(0.05)
            await client.disconnect()

        assert ("타임스탬프 테스트", True) in transcripts
        assert utt_ends == [True]

    async def test_is_connected_property(self):
        client = STTClient(api_key="test")
        assert client.is_connected is False

        ws = _make_ws_mock()
        with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
            await client.connect()
            assert client.is_connected is True
            await client.disconnect()
            assert client.is_connected is False

    async def test_connect_retries_on_failure(self):
        """2 OSError then success → call_count == 3, sleep called with 0.5 and 1.0."""
        ws = _make_ws_mock()
        mock_connect = AsyncMock(
            side_effect=[OSError("fail"), OSError("fail"), ws]
        )
        with patch("prot.stt.websockets.connect", mock_connect), \
             patch("prot.stt.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            client = STTClient(api_key="test")
            await client.connect()
            assert mock_connect.call_count == 3
            assert client.is_connected is True
            assert mock_sleep.await_args_list[0].args == (0.5,)
            assert mock_sleep.await_args_list[1].args == (1.0,)
            await client.disconnect()

    async def test_connect_exhausts_retries(self):
        """All attempts OSError → _ws is None, call_count == 4."""
        mock_connect = AsyncMock(side_effect=OSError("fail"))
        with patch("prot.stt.websockets.connect", mock_connect), \
             patch("prot.stt.asyncio.sleep", new_callable=AsyncMock):
            client = STTClient(api_key="test")
            await client.connect()
            assert client._ws is None
            assert client.is_connected is False
            assert mock_connect.call_count == 4

    async def test_connect_session_timeout(self):
        """recv hangs → wait_for timeout → _ws is None."""
        ws = AsyncMock()
        ws.close = AsyncMock()
        # recv never resolves
        never = asyncio.get_event_loop().create_future()
        ws.recv = AsyncMock(return_value=never)

        mock_connect = AsyncMock(return_value=ws)
        with patch("prot.stt.websockets.connect", mock_connect), \
             patch("prot.stt.asyncio.sleep", new_callable=AsyncMock):
            client = STTClient(api_key="test")
            await client.connect()
            assert client._ws is None
            assert client.is_connected is False

    async def test_connect_reuses_open_websocket(self):
        """connect() should reuse existing open WebSocket instead of reconnecting."""
        client = STTClient(api_key="test")
        mock_ws = AsyncMock()
        mock_ws.open = True
        client._ws = mock_ws
        client._recv_task = MagicMock()
        client._recv_task.done.return_value = False

        with patch("prot.stt.websockets.connect") as mock_connect:
            await client.connect()
            mock_connect.assert_not_called()

    async def test_send_audio_disconnect_failure_clears_recv_task(self):
        """If disconnect() raises in send_audio fallback, _recv_task should be cleared."""
        client = STTClient(api_key="test")
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock(side_effect=Exception("send failed"))
        client._ws = mock_ws
        mock_task = MagicMock()
        client._recv_task = mock_task

        # Make disconnect itself raise
        original_disconnect = client.disconnect
        client.disconnect = AsyncMock(side_effect=Exception("disconnect failed"))

        await client.send_audio(b"\x00" * 100)
        assert client._ws is None
        assert client._recv_task is None
