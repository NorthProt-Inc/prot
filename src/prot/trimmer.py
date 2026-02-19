"""Token-budget context trimmer for conversation messages.

Trims conversation history to fit within a token budget, using
Anthropic's count_tokens API for exact counting on the first call
and response usage data for subsequent tool-loop iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prot.config import settings
from prot.logging import get_logger
from prot.processing import is_tool_result_message

if TYPE_CHECKING:
    from prot.llm import LLMClient

logger = get_logger(__name__)


def _estimate_tokens(char_count: int) -> int:
    """Rough token estimate: ~4 chars per token."""
    return char_count // 4 + 1


def _message_text_len(msg: dict) -> int:
    """Estimate character length of a message's content."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += len(str(block.get("content", "")))
                total += len(str(block.get("text", "")))
            else:
                total += len(str(block))
        return total
    return len(str(content))


class TokenBudgetTrimmer:
    """Trim messages to fit within a token budget.

    First call: uses count_tokens() API for exact count.
    Subsequent calls (tool loop): uses previous response's usage.input_tokens
    to avoid additional API calls.
    """

    def __init__(
        self,
        llm: LLMClient,
        system: list[dict],
        tools: list[dict] | None,
        budget: int,
        tool_result_max_chars: int | None = None,
    ) -> None:
        self._llm = llm
        self._system = system
        self._tools = tools
        self._budget = budget
        self._tool_result_max_chars = (
            tool_result_max_chars
            if tool_result_max_chars is not None
            else settings.context_tool_result_max_chars
        )
        self._last_input_tokens: int | None = None
        self._last_message_count: int = 0

    async def fit(self, messages: list[dict]) -> list[dict]:
        """Return messages trimmed to fit within token budget."""
        if not messages:
            return messages

        messages = self._truncate_tool_results(messages)

        token_count = await self._count(messages)

        # Trim oldest exchanges until within budget
        while token_count > self._budget and len(messages) > 2:
            messages = self._remove_oldest_exchange(messages)
            token_count = await self._count(messages)

        # Ensure valid boundary (no orphaned tool_result at start)
        messages = self._fix_boundary(messages)

        return messages

    def update_overhead(self, usage) -> None:
        """Store usage.input_tokens from a completed response."""
        if usage is not None:
            self._last_input_tokens = usage.input_tokens

    async def _count(self, messages: list[dict]) -> int:
        """Count tokens — API on first call, estimate on subsequent."""
        if self._last_input_tokens is not None:
            # Estimate: known overhead + delta from new messages
            delta_chars = sum(
                _message_text_len(m)
                for m in messages[self._last_message_count:]
            )
            estimate = self._last_input_tokens + _estimate_tokens(delta_chars)
            self._last_message_count = len(messages)
            return estimate

        # First call — use exact API count
        try:
            count = await self._llm.count_tokens(
                system=self._system,
                tools=self._tools,
                messages=messages,
            )
            self._last_message_count = len(messages)
            return count
        except Exception:
            logger.warning("count_tokens failed, using heuristic", exc_info=True)
            total_chars = sum(_message_text_len(m) for m in messages)
            return _estimate_tokens(total_chars)

    def _truncate_tool_results(self, messages: list[dict]) -> list[dict]:
        """Truncate long tool_result content blocks."""
        max_chars = self._tool_result_max_chars
        result = []
        for msg in messages:
            if not isinstance(msg.get("content"), list):
                result.append(msg)
                continue
            new_blocks = []
            mutated = False
            for block in msg["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and len(block["content"]) > max_chars
                ):
                    new_block = dict(block)
                    new_block["content"] = (
                        block["content"][:max_chars] + " [truncated]"
                    )
                    new_blocks.append(new_block)
                    mutated = True
                else:
                    new_blocks.append(block)
            if mutated:
                result.append({"role": msg["role"], "content": new_blocks})
            else:
                result.append(msg)
        return result

    @staticmethod
    def _remove_oldest_exchange(messages: list[dict]) -> list[dict]:
        """Remove the oldest user+assistant exchange from the front."""
        if len(messages) <= 2:
            return messages
        return messages[2:]

    @staticmethod
    def _fix_boundary(messages: list[dict]) -> list[dict]:
        """Ensure messages start at a valid user message (not orphaned tool_result)."""
        while messages and (
            messages[0]["role"] != "user"
            or is_tool_result_message(messages[0])
        ):
            messages = messages[1:]
        return messages
