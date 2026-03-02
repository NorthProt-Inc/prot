"""Home Assistant integration — conversation API delegation."""

from __future__ import annotations

import httpx

from prot.logging import get_logger, logged

logger = get_logger(__name__)

_HASS_TIMEOUT = httpx.Timeout(15.0)

_TOOL_SCHEMA: dict = {
    "name": "hass_request",
    "description": (
        "Send a natural language command to the Home Assistant agent. "
        "Use for any smart home task: device control, state queries, "
        "automation triggers, scene activation, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Natural language command for Home Assistant. "
                    "Be specific with device names and values. "
                    "Example: '조명 밝기 40%, 색온도 2700K로 변경'"
                ),
            }
        },
        "required": ["command"],
    },
}


class HassAgent:
    """Delegate smart home commands to HA's conversation agent."""

    def __init__(self, url: str, token: str, agent_id: str = "") -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._agent_id = agent_id
        self._client = httpx.AsyncClient(timeout=_HASS_TIMEOUT)

    @logged(slow_ms=2000, log_args=True)
    async def request(self, command: str) -> str:
        """Send command to HA conversation agent, return response text."""
        try:
            resp = await self._client.post(
                f"{self._url}/api/conversation/process",
                headers={"Authorization": f"Bearer {self._token}"},
                json={
                    "text": command,
                    "language": "ko",
                    **({"agent_id": self._agent_id} if self._agent_id else {}),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["response"]["speech"]["plain"]["speech"]
        except httpx.ConnectError:
            logger.warning("HASS connection failed")
            return "Home Assistant에 연결할 수 없습니다"
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                logger.warning("HASS auth failed")
                return "Home Assistant 인증 실패"
            logger.warning("HASS HTTP error", status=status)
            return f"Home Assistant 오류 (HTTP {status})"
        except httpx.TimeoutException:
            logger.warning("HASS request timed out")
            return "Home Assistant 응답 시간 초과"

    def build_tool(self) -> dict:
        """Return single tool schema for Claude."""
        return dict(_TOOL_SCHEMA)

    async def close(self) -> None:
        """Close the httpx client."""
        await self._client.aclose()
