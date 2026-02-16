import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prot.llm import LLMClient


@pytest.mark.asyncio
class TestLLMClient:
    async def test_stream_response_yields_text(self):
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream._events = [
            MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="안녕")),
            MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="하세요.")),
        ]
        mock_stream.__anext__ = AsyncMock(side_effect=[
            mock_stream._events[0],
            mock_stream._events[1],
            StopAsyncIteration,
        ])

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            chunks = []
            async for chunk in client.stream_response(
                system_blocks=[{"type": "text", "text": "test"}],
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            ):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0] == "안녕"

    async def test_cancel_stops_stream(self):
        client = LLMClient(api_key="test")
        client.cancel()
        assert client._cancelled is True
