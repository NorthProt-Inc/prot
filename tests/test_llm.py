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

    async def test_stream_uses_ga_api(self):
        """stream_response uses GA messages.stream (not beta), no compaction."""
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_stream.get_final_message = AsyncMock(
            return_value=MagicMock(content=[MagicMock(text="ok")])
        )

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[{"type": "text", "text": "test"}],
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass

            call_kwargs = mock_client.messages.stream.call_args.kwargs
            assert "betas" not in call_kwargs
            assert "context_management" not in call_kwargs
            assert call_kwargs["thinking"] == {"type": "adaptive"}

    async def test_last_response_content_captured(self):
        mock_content = [MagicMock(text="response")]
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_stream.get_final_message = AsyncMock(
            return_value=MagicMock(content=mock_content)
        )

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[], tools=[], messages=[],
            ):
                pass

            assert client.last_response_content is mock_content

    async def test_last_response_content_none_on_error(self):
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_stream.get_final_message = AsyncMock(side_effect=RuntimeError("fail"))

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[], tools=[], messages=[],
            ):
                pass

            assert client.last_response_content is None

    async def test_cancel_stops_stream(self):
        client = LLMClient(api_key="test")
        client.cancel()
        assert client._cancelled is True

    async def test_execute_tool_unknown_returns_error(self):
        client = LLMClient(api_key="test")
        result = await client.execute_tool("unknown_tool", {})
        assert "error" in result
        assert "unknown_tool" in result["error"].lower()

    async def test_stream_passes_empty_tools_as_list(self):
        """Empty tools list should be passed as-is, not converted to None."""
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_stream.get_final_message = AsyncMock(
            return_value=MagicMock(content=[])
        )

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[], tools=[], messages=[],
            ):
                pass

            call_kwargs = mock_client.messages.stream.call_args.kwargs
            assert call_kwargs["tools"] == []

    async def test_stream_passes_none_tools_as_none(self):
        """None tools should be passed as None."""
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_stream.get_final_message = AsyncMock(
            return_value=MagicMock(content=[])
        )

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[], tools=None, messages=[],
            ):
                pass

            call_kwargs = mock_client.messages.stream.call_args.kwargs
            assert call_kwargs["tools"] is None


class TestToolDetection:
    def test_get_tool_use_blocks_extracts_tools(self):
        client = LLMClient.__new__(LLMClient)
        tool = MagicMock(type="tool_use")
        text = MagicMock(type="text")
        client._last_response_content = [text, tool]
        assert client.get_tool_use_blocks() == [tool]

    def test_get_tool_use_blocks_empty_when_no_tools(self):
        client = LLMClient.__new__(LLMClient)
        client._last_response_content = [MagicMock(type="text")]
        assert client.get_tool_use_blocks() == []

    def test_get_tool_use_blocks_empty_when_none(self):
        client = LLMClient.__new__(LLMClient)
        client._last_response_content = None
        assert client.get_tool_use_blocks() == []


class TestStreamResponseResetContent:
    """stream_response resets _last_response_content before streaming."""

    async def test_last_response_content_reset_before_stream(self):
        """_last_response_content is None if stream fails before completion."""
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
