import httpx
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prot.tts import TTSClient


@pytest.mark.asyncio
class TestTTSClient:
    async def test_stream_audio_yields_bytes(self):
        mock_response = AsyncMock()
        mock_response.__aiter__ = lambda self: self
        mock_response.__anext__ = AsyncMock(
            side_effect=[b"\x00" * 1024, b"\x00" * 512, StopAsyncIteration]
        )

        with patch("prot.tts.AsyncElevenLabs") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.text_to_speech.stream = MagicMock(
                return_value=mock_response
            )

            tts = TTSClient(api_key="test")
            chunks = []
            async for chunk in tts.stream_audio("테스트 문장"):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert all(isinstance(c, bytes) for c in chunks)

    async def test_stream_audio_passes_settings(self):
        mock_response = AsyncMock()
        mock_response.__aiter__ = lambda self: self
        mock_response.__anext__ = AsyncMock(side_effect=[StopAsyncIteration])

        with patch("prot.tts.AsyncElevenLabs") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.text_to_speech.stream = MagicMock(
                return_value=mock_response
            )

            tts = TTSClient(api_key="test")
            async for _ in tts.stream_audio("hello"):
                pass

            mock_client.text_to_speech.stream.assert_called_once()
            call_kwargs = mock_client.text_to_speech.stream.call_args
            assert call_kwargs.kwargs["text"] == "hello"
            assert "voice_id" in call_kwargs.kwargs
            assert "model_id" in call_kwargs.kwargs
            assert "output_format" in call_kwargs.kwargs

    async def test_stream_audio_skips_non_bytes(self):
        mock_response = AsyncMock()
        mock_response.__aiter__ = lambda self: self
        mock_response.__anext__ = AsyncMock(
            side_effect=[b"\x00" * 100, "not bytes", b"\xff" * 100, StopAsyncIteration]
        )

        with patch("prot.tts.AsyncElevenLabs") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.text_to_speech.stream = MagicMock(
                return_value=mock_response
            )

            tts = TTSClient(api_key="test")
            chunks = []
            async for chunk in tts.stream_audio("test"):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert all(isinstance(c, bytes) for c in chunks)

    async def test_flush_clears_state(self):
        with patch("prot.tts.AsyncElevenLabs"):
            tts = TTSClient(api_key="test")
            tts.flush()
            assert tts._cancelled is True

    async def test_flush_stops_active_stream(self):
        chunk_count = 0

        async def slow_iter():
            nonlocal chunk_count
            for i in range(10):
                chunk_count += 1
                yield b"\x00" * 100

        with patch("prot.tts.AsyncElevenLabs") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.text_to_speech.stream = MagicMock(
                return_value=slow_iter()
            )

            tts = TTSClient(api_key="test")
            chunks = []
            async for chunk in tts.stream_audio("test"):
                chunks.append(chunk)
                if len(chunks) == 2:
                    tts.flush()

            assert len(chunks) < 10

    async def test_stream_audio_resets_cancelled(self):
        mock_response = AsyncMock()
        mock_response.__aiter__ = lambda self: self
        mock_response.__anext__ = AsyncMock(
            side_effect=[b"\x00" * 100, StopAsyncIteration]
        )

        with patch("prot.tts.AsyncElevenLabs") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.text_to_speech.stream = MagicMock(
                return_value=mock_response
            )

            tts = TTSClient(api_key="test")
            tts._cancelled = True

            chunks = []
            async for chunk in tts.stream_audio("test"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert tts._cancelled is False

    async def test_stream_audio_handles_connect_error(self):
        """httpx.ConnectError → yields [], no exception."""
        with patch("prot.tts.AsyncElevenLabs") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            async def _raise_connect(*a, **kw):
                raise httpx.ConnectError("connection refused")
                yield  # make it a generator  # noqa: E501

            mock_client.text_to_speech.stream = MagicMock(
                return_value=_raise_connect()
            )

            tts = TTSClient(api_key="test")
            chunks = []
            async for chunk in tts.stream_audio("test"):
                chunks.append(chunk)

            assert chunks == []

    async def test_stream_audio_handles_network_error(self):
        """OSError → yields [], no exception."""
        with patch("prot.tts.AsyncElevenLabs") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            async def _raise_os(*a, **kw):
                raise OSError("network unreachable")
                yield  # make it a generator  # noqa: E501

            mock_client.text_to_speech.stream = MagicMock(
                return_value=_raise_os()
            )

            tts = TTSClient(api_key="test")
            chunks = []
            async for chunk in tts.stream_audio("test"):
                chunks.append(chunk)

            assert chunks == []
