from anthropic import AsyncAnthropic

from prot.config import settings
from prot.logging import get_logger, logged

logger = get_logger(__name__)

_BETAS = ["compact-2026-01-12", "context-management-2025-06-27"]


def _build_context_management() -> dict:
    """Build context_management.edits from settings."""
    compact_edit = {
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": settings.compaction_trigger},
        "instructions": (
            "Summarize this conversation preserving: "
            "(1) temporal markers — when topics changed, time references by either party, "
            "day boundaries, and session gaps with approximate timestamps; "
            "(2) key facts, decisions, and user preferences; "
            "(3) emotional context and relationship dynamics."
        ),
    }
    if settings.pause_after_compaction:
        compact_edit["pause_after_compaction"] = True

    return {
        "edits": [
            {
                "type": "clear_thinking_20251015",
                "keep": {"type": "thinking_turns", "value": settings.thinking_keep_turns},
            },
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": settings.tool_clear_trigger},
                "keep": {"type": "tool_uses", "value": settings.tool_clear_keep},
            },
            compact_edit,
        ],
    }


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self._cancelled = False
        self._last_response_content = None
        self._last_usage = None
        self._last_compaction_summary = None

    @logged(slow_ms=2000, log_args=True)
    async def stream_response(
        self,
        system_blocks: list[dict],
        tools: list[dict] | None,
        messages: list[dict],
    ):
        """Stream text deltas from Claude via Beta API with server-side context management.

        Context management (applied server-side, in order):
          1. Thinking block clearing — keep last N turns
          2. Tool result clearing — clear old tool results above trigger threshold
          3. Compaction — summarize conversation above trigger threshold
        """
        self._cancelled = False
        self._last_response_content = None  # prevent stale tool blocks on generator abandonment
        logger.info("Streaming", model=settings.claude_model)

        async with self._client.beta.messages.stream(
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": settings.claude_effort},
            system=system_blocks,
            tools=tools,
            messages=messages,
            betas=_BETAS,
            context_management=_build_context_management(),
        ) as stream:
            async for event in stream:
                if self._cancelled:
                    break
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield event.delta.text

        try:
            final = await stream.get_final_message()
            self._last_response_content = final.content
            self._last_usage = final.usage

            # Detect compaction events
            self._last_compaction_summary = None
            if getattr(final, "stop_reason", None) == "compaction":
                for block in final.content:
                    if getattr(block, "type", None) == "compaction":
                        self._last_compaction_summary = getattr(block, "content", None)
                        break
        except Exception:
            self._last_response_content = None
            self._last_usage = None

    @property
    def last_usage(self):
        """Token usage from the last streamed response."""
        return self._last_usage

    @property
    def last_compaction_summary(self) -> str | None:
        """Compaction summary from last stream, if compaction occurred."""
        return self._last_compaction_summary

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
        await self._client.close()

