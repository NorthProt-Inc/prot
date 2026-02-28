"""Tests for HassAgent — HA conversation API delegation."""

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prot.hass import HassAgent


class TestHassAgentInit:
    def test_strips_trailing_slash(self):
        agent = HassAgent("http://hass:8123/", "token")
        assert agent._url == "http://hass:8123"

    def test_stores_token(self):
        agent = HassAgent("http://hass:8123", "my-token")
        assert agent._token == "my-token"


class TestHassAgentBuildTool:
    def test_returns_valid_tool_schema(self):
        agent = HassAgent("http://hass:8123", "token")
        tool = agent.build_tool()
        assert tool["name"] == "hass_request"
        assert "input_schema" in tool
        props = tool["input_schema"]["properties"]
        assert "command" in props
        assert tool["input_schema"]["required"] == ["command"]

    def test_schema_has_description(self):
        agent = HassAgent("http://hass:8123", "token")
        tool = agent.build_tool()
        assert "description" in tool
        assert len(tool["description"]) > 0


class TestHassAgentRequest:
    async def test_success_returns_speech(self):
        agent = HassAgent("http://hass:8123", "token")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": {
                "speech": {
                    "plain": {"speech": "거실 조명을 켰습니다"}
                }
            }
        }
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)

        result = await agent.request("거실 조명 켜줘")

        assert result == "거실 조명을 켰습니다"
        call_args = agent._client.post.call_args
        assert "/api/conversation/process" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["text"] == "거실 조명 켜줘"
        assert body["language"] == "ko"

    async def test_posts_with_auth_header(self):
        agent = HassAgent("http://hass:8123", "my-token")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": {"speech": {"plain": {"speech": "ok"}}}
        }
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)

        await agent.request("test")

        headers = agent._client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-token"

    async def test_connection_error_returns_message(self):
        agent = HassAgent("http://hass:8123", "token")
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await agent.request("test")
        assert "연결할 수 없습니다" in result

    async def test_auth_error_returns_message(self):
        agent = HassAgent("http://hass:8123", "token")
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)

        result = await agent.request("test")
        assert "인증 실패" in result

    async def test_timeout_returns_message(self):
        agent = HassAgent("http://hass:8123", "token")
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("timed out")
        )

        result = await agent.request("test")
        assert "시간 초과" in result

    async def test_generic_http_error_returns_message(self):
        agent = HassAgent("http://hass:8123", "token")
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)

        result = await agent.request("test")
        assert "오류" in result or "실패" in result


class TestHassAgentClose:
    async def test_close_calls_aclose(self):
        agent = HassAgent("http://hass:8123", "token")
        agent._client = AsyncMock()
        await agent.close()
        agent._client.aclose.assert_awaited_once()
