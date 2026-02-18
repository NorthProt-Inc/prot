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

    async def test_execute_hass_invalid_entity_id_returns_error(self):
        client = LLMClient(api_key="test")
        result = await client.execute_tool(
            "home_assistant", {"action": "get_state", "entity_id": "../../bad"}
        )
        assert "error" in result
        assert "Invalid entity_id" in result["error"]

    async def test_execute_hass_missing_entity_id_returns_error(self):
        client = LLMClient(api_key="test")
        result = await client.execute_tool(
            "home_assistant", {"action": "get_state"}
        )
        assert "error" in result

    async def test_execute_hass_get_state_success(self):
        client = LLMClient(api_key="test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"state": "on"}

        client._hass_client = AsyncMock()
        client._hass_client.get = AsyncMock(return_value=mock_response)

        result = await client.execute_tool(
            "home_assistant",
            {"action": "get_state", "entity_id": "light.living_room"},
        )
        assert result == {"state": "on"}

    async def test_execute_hass_http_error_returns_error(self):
        client = LLMClient(api_key="test")
        mock_response = MagicMock()
        mock_response.status_code = 401

        client._hass_client = AsyncMock()
        client._hass_client.get = AsyncMock(return_value=mock_response)

        result = await client.execute_tool(
            "home_assistant",
            {"action": "get_state", "entity_id": "light.living_room"},
        )
        assert "error" in result
        assert "401" in result["error"]

    async def test_close_closes_hass_client(self):
        client = LLMClient(api_key="test")
        client._hass_client = AsyncMock()
        await client.close()
        client._hass_client.aclose.assert_awaited_once()


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
