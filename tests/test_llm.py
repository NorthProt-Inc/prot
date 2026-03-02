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
            mock_client.beta.messages.stream.return_value = mock_stream

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

    async def test_stream_uses_beta_api_with_context_management(self):
        """stream_response uses beta messages.stream with compaction + context editing."""
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
            mock_client.beta.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[{"type": "text", "text": "test"}],
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass

            mock_client.beta.messages.stream.assert_called_once()
            call_kwargs = mock_client.beta.messages.stream.call_args.kwargs
            assert "compact-2026-01-12" in call_kwargs["betas"]
            assert "context-management-2025-06-27" in call_kwargs["betas"]
            assert "context_management" in call_kwargs
            edits = call_kwargs["context_management"]["edits"]
            edit_types = [e["type"] for e in edits]
            assert edit_types == [
                "clear_thinking_20251015",
                "clear_tool_uses_20250919",
                "compact_20260112",
            ]
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
            mock_client.beta.messages.stream.return_value = mock_stream

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
            mock_client.beta.messages.stream.return_value = mock_stream

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
            mock_client.beta.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[], tools=[], messages=[],
            ):
                pass

            call_kwargs = mock_client.beta.messages.stream.call_args.kwargs
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
            mock_client.beta.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[], tools=None, messages=[],
            ):
                pass

            call_kwargs = mock_client.beta.messages.stream.call_args.kwargs
            assert call_kwargs["tools"] is None


class TestLastUsage:
    async def test_last_usage_captured(self):
        mock_usage = MagicMock(input_tokens=150, output_tokens=50)
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_stream.get_final_message = AsyncMock(
            return_value=MagicMock(content=[], usage=mock_usage)
        )

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            assert client.last_usage is None
            async for _ in client.stream_response(
                system_blocks=[], tools=[], messages=[],
            ):
                pass

            assert client.last_usage is mock_usage
            assert client.last_usage.input_tokens == 150

    async def test_last_usage_none_on_error(self):
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_stream.get_final_message = AsyncMock(side_effect=RuntimeError("fail"))

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[], tools=[], messages=[],
            ):
                pass

            assert client.last_usage is None



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


class TestCompactionDetection:
    async def test_compaction_edit_includes_pause(self):
        """Compaction edit should include pause_after_compaction when enabled."""
        with patch("prot.llm.settings") as ms:
            ms.thinking_keep_turns = 2
            ms.tool_clear_trigger = 30000
            ms.tool_clear_keep = 3
            ms.compaction_trigger = 50000
            ms.pause_after_compaction = True

            from prot.llm import _build_context_management
            cm = _build_context_management()
            compact_edit = cm["edits"][2]
            assert compact_edit["type"] == "compact_20260112"
            assert compact_edit["pause"] is True

    async def test_last_compaction_summary_initially_none(self):
        client = LLMClient.__new__(LLMClient)
        client._last_compaction_summary = None
        assert client.last_compaction_summary is None

    async def test_stop_reason_compaction_detected(self):
        """When stop_reason is 'compaction', last_compaction_summary should be populated."""
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)

        # Simulate compaction stop reason with compaction content block
        compaction_block = MagicMock()
        compaction_block.type = "compaction"
        compaction_block.summary = "User discussed Python debugging techniques."
        final_msg = MagicMock()
        final_msg.content = [compaction_block]
        final_msg.stop_reason = "compaction"
        final_msg.usage = MagicMock()
        mock_stream.get_final_message = AsyncMock(return_value=final_msg)

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response([], None, []):
                pass

            assert client.last_compaction_summary == "User discussed Python debugging techniques."


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
        client._client.beta.messages.stream = MagicMock(
            side_effect=RuntimeError("connection failed")
        )

        with pytest.raises(RuntimeError):
            async for _ in client.stream_response([], None, []):
                pass

        # Should be reset to None, not stale tool_use blocks
        assert client._last_response_content is None
