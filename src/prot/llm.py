import re

import httpx
from anthropic import AsyncAnthropic

from prot.config import settings

_ENTITY_ID_PATTERN = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")
_HASS_TIMEOUT = httpx.Timeout(10.0)


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self._cancelled = False
        self._active_stream = None

    async def stream_response(
        self,
        system_blocks: list[dict],
        tools: list[dict],
        messages: list[dict],
    ):
        """Stream text deltas from Claude. Yields str chunks.

        system_blocks order: [persona (cached), rag (cached), dynamic (NOT cached)]
        """
        self._cancelled = False

        async with self._client.messages.stream(
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": settings.claude_effort},
            system=system_blocks,
            tools=tools if tools else None,
            messages=messages,
        ) as stream:
            self._active_stream = stream
            async for event in stream:
                if self._cancelled:
                    break
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield event.delta.text

        self._active_stream = None

    def cancel(self) -> None:
        """Cancel the active stream."""
        self._cancelled = True

    async def execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool call from Claude. Returns tool result."""
        if tool_name == "home_assistant":
            return await self._execute_hass(tool_input)
        return {"error": f"Unknown tool: {tool_name}"}

    async def _execute_hass(self, tool_input: dict) -> dict:
        """Execute Home Assistant API call."""
        action = tool_input.get("action")
        entity_id = tool_input.get("entity_id", "")

        if not _ENTITY_ID_PATTERN.match(entity_id):
            return {"error": f"Invalid entity_id: {entity_id}"}

        async with httpx.AsyncClient(timeout=_HASS_TIMEOUT) as http:
            headers = {"Authorization": f"Bearer {settings.hass_token}"}
            if action == "get_state":
                r = await http.get(
                    f"{settings.hass_url}/api/states/{entity_id}",
                    headers=headers,
                )
                if r.status_code != 200:
                    return {"error": f"HASS returned {r.status_code}"}
                return r.json()
            elif action == "call_service":
                domain, service = entity_id.split(".", 1)
                r = await http.post(
                    f"{settings.hass_url}/api/services/{domain}/{service}",
                    headers=headers,
                    json=tool_input.get("service_data", {}),
                )
                if r.status_code != 200:
                    return {"error": f"HASS returned {r.status_code}"}
                return r.json()
        return {"error": "Invalid action"}
