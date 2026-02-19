from anthropic import AsyncAnthropic

from prot.config import settings
from prot.logging import get_logger, logged

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self._cancelled = False
        self._active_stream = None
        self._last_response_content = None

    @logged(slow_ms=2000, log_args=True)
    async def stream_response(
        self,
        system_blocks: list[dict],
        tools: list[dict] | None,
        messages: list[dict],
    ):
        """Stream text deltas from Claude. Yields str chunks.

        system_blocks order: [persona (cached), rag (cached), dynamic (NOT cached)]
        """
        self._cancelled = False
        self._last_response_content = None  # prevent stale tool blocks on generator abandonment
        logger.info("Streaming", model=settings.claude_model)

        async with self._client.messages.stream(
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": settings.claude_effort},
            system=system_blocks,
            tools=tools,
            messages=messages,
            # --- Compaction (Opus 4.6 only, re-enable when Sonnet supports it) ---
            # betas=["compact-2026-01-12"],
            # context_management={
            #     "edits": [{"type": "compact_20260112"}],
            # },
        ) as stream:
            self._active_stream = stream
            async for event in stream:
                if self._cancelled:
                    break
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield event.delta.text

        self._active_stream = None

        try:
            final = await stream.get_final_message()
            self._last_response_content = final.content
        except Exception:
            self._last_response_content = None

    @property
    def last_response_content(self):
        """Full response content blocks from last stream."""
        return self._last_response_content

    def get_tool_use_blocks(self) -> list:
        """Extract tool_use blocks from last response."""
        if not self._last_response_content:
            return []
        return [b for b in self._last_response_content if getattr(b, "type", None) == "tool_use"]

    def cancel(self) -> None:
        """Cancel the active stream."""
        self._cancelled = True

    async def close(self) -> None:
        """Close persistent HTTP clients."""

    async def execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool call from Claude. Returns tool result."""
        logger.info("Tool call", tool=tool_name)
        return {"error": f"Unknown tool: {tool_name}"}
