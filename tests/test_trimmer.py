"""Tests for TokenBudgetTrimmer — token-budget context management."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from prot.trimmer import TokenBudgetTrimmer


def _msg(role: str, content: str | list) -> dict:
    return {"role": role, "content": content}


def _tool_result(tool_use_id: str, content: str) -> dict:
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    ]}


class TestFitUnderBudget:
    """Messages that fit within budget are returned as-is."""

    async def test_returns_all_when_under_budget(self):
        llm = MagicMock()
        llm.count_tokens = AsyncMock(return_value=1000)

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
        )
        messages = [_msg("user", "hi"), _msg("assistant", "hello")]
        result = await trimmer.fit(messages)
        assert result == messages

    async def test_empty_messages(self):
        llm = MagicMock()
        llm.count_tokens = AsyncMock(return_value=100)

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
        )
        result = await trimmer.fit([])
        assert result == []


class TestFitOverBudget:
    """Messages exceeding budget are trimmed from the front."""

    async def test_trims_oldest_exchanges(self):
        call_count = 0

        async def mock_count(system, tools, messages):
            nonlocal call_count
            call_count += 1
            # First call: over budget; second call: under budget
            return 40000 if call_count == 1 else 20000

        llm = MagicMock()
        llm.count_tokens = mock_count

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
        )
        messages = [
            _msg("user", "old msg"),
            _msg("assistant", "old resp"),
            _msg("user", "new msg"),
            _msg("assistant", "new resp"),
        ]
        result = await trimmer.fit(messages)
        # Should have removed first exchange
        assert len(result) == 2
        assert result[0]["content"] == "new msg"

    async def test_always_keeps_last_user_message(self):
        llm = MagicMock()
        # Always over budget
        llm.count_tokens = AsyncMock(return_value=50000)

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
        )
        messages = [
            _msg("user", "only msg"),
            _msg("assistant", "only resp"),
        ]
        result = await trimmer.fit(messages)
        # Must keep at least the last user message
        assert len(result) >= 1
        assert any(m["role"] == "user" for m in result)

    async def test_skips_orphan_tool_result_after_trim(self):
        call_count = 0

        async def mock_count(system, tools, messages):
            nonlocal call_count
            call_count += 1
            return 40000 if call_count == 1 else 20000

        llm = MagicMock()
        llm.count_tokens = mock_count

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
        )
        messages = [
            _msg("user", "old"),
            _msg("assistant", "old resp"),
            _tool_result("t1", "some result"),
            _msg("assistant", "based on tool"),
            _msg("user", "new"),
            _msg("assistant", "new resp"),
        ]
        result = await trimmer.fit(messages)
        # After trim, should not start with tool_result
        assert result[0]["role"] == "user"
        if isinstance(result[0]["content"], list):
            assert not all(
                b.get("type") == "tool_result" for b in result[0]["content"]
            )


class TestToolResultTruncation:
    """Long tool_result content is truncated to save budget."""

    async def test_truncates_long_tool_result(self):
        long_content = "x" * 5000

        call_count = 0

        async def mock_count(system, tools, messages):
            nonlocal call_count
            call_count += 1
            # Over budget on first call, under after truncation
            return 40000 if call_count <= 1 else 20000

        llm = MagicMock()
        llm.count_tokens = mock_count

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
            tool_result_max_chars=200,
        )
        messages = [
            _msg("user", "query"),
            _msg("assistant", "checking"),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": long_content}
            ]},
            _msg("assistant", "result"),
        ]
        result = await trimmer.fit(messages)

        # Find the tool_result and check it was truncated
        for msg in result:
            if isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result" and block.get("tool_use_id") == "t1":
                        assert len(block["content"]) <= 250  # max_chars + marker
                        assert "[truncated]" in block["content"]

    async def test_preserves_tool_use_id(self):
        llm = MagicMock()
        llm.count_tokens = AsyncMock(return_value=40000)

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
            tool_result_max_chars=100,
        )
        messages = [
            _msg("user", "q"),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t99", "content": "x" * 500}
            ]},
            _msg("assistant", "r"),
        ]
        result = await trimmer.fit(messages)

        for msg in result:
            if isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result":
                        assert block["tool_use_id"] == "t99"

    async def test_non_string_content_not_truncated(self):
        """web_search encrypted content (non-string) should not be truncated."""
        llm = MagicMock()
        llm.count_tokens = AsyncMock(return_value=1000)

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
            tool_result_max_chars=100,
        )
        encrypted_content = [{"type": "encrypted", "data": "abc" * 1000}]
        messages = [
            _msg("user", "q"),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "ws1", "content": encrypted_content}
            ]},
            _msg("assistant", "r"),
        ]
        result = await trimmer.fit(messages)

        # Content should be unchanged (not truncated)
        for msg in result:
            if isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result" and block.get("tool_use_id") == "ws1":
                        assert block["content"] is encrypted_content


class TestUpdateOverhead:
    """update_overhead() uses usage.input_tokens for subsequent calls."""

    async def test_second_call_skips_count_tokens(self):
        llm = MagicMock()
        llm.count_tokens = AsyncMock(return_value=5000)

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
        )
        messages = [_msg("user", "hi"), _msg("assistant", "hello")]

        # First call — uses count_tokens
        await trimmer.fit(messages)
        assert llm.count_tokens.await_count == 1

        # Simulate response usage
        usage = MagicMock(input_tokens=5000)
        trimmer.update_overhead(usage)

        # Second call — should NOT call count_tokens again
        messages.append(_msg("user", [
            {"type": "tool_result", "tool_use_id": "t1", "content": "result"}
        ]))
        messages.append(_msg("assistant", "noted"))
        await trimmer.fit(messages)
        assert llm.count_tokens.await_count == 1  # still 1, not 2


class TestCountTokensFallback:
    """Fallback to heuristic when count_tokens API fails."""

    async def test_fallback_on_api_error(self):
        llm = MagicMock()
        llm.count_tokens = AsyncMock(side_effect=Exception("API down"))

        trimmer = TokenBudgetTrimmer(
            llm=llm, system=[], tools=[], budget=30000,
        )
        messages = [_msg("user", "short")]
        # Should not raise — falls back to heuristic
        result = await trimmer.fit(messages)
        assert len(result) >= 1
