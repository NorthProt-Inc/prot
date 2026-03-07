# tests/test_engine.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass

from prot.engine import ConversationEngine, ResponseResult, BusyError, ToolIterationMarker


def _make_engine(**overrides):
    """Create engine with mocked dependencies."""
    ctx = MagicMock()
    ctx.build_system_blocks.return_value = [{"type": "text", "text": "persona"}]
    ctx.build_tools.return_value = []
    ctx.get_recent_messages.return_value = [{"role": "user", "content": "hello"}]
    ctx.add_message = MagicMock()
    ctx.get_messages.return_value = []

    llm = MagicMock()
    llm._cancelled = False
    llm.cancel = MagicMock(side_effect=lambda: setattr(llm, "_cancelled", True))
    llm.get_tool_use_blocks.return_value = []
    llm.last_response_content = [{"type": "text", "text": "response"}]
    llm.last_compaction_summary = None
    llm.close = AsyncMock()

    defaults = dict(ctx=ctx, llm=llm)
    defaults.update(overrides)
    return ConversationEngine(**defaults)


class TestRespond:
    async def test_yields_text_chunks(self):
        engine = _make_engine()

        async def fake_stream(*a, **kw):
            for word in ["Hello", " world"]:
                yield word

        engine._llm.stream_response = fake_stream

        chunks = [c async for c in engine.respond()]

        assert chunks == ["Hello", " world"]
        assert engine.last_result.full_text == "Hello world"
        assert engine.last_result.interrupted is False

    async def test_commits_response_to_context(self):
        engine = _make_engine()

        async def fake_stream(*a, **kw):
            yield "hi"

        engine._llm.stream_response = fake_stream

        async for _ in engine.respond():
            pass

        engine._ctx.add_message.assert_called_once_with(
            "assistant", engine._llm.last_response_content
        )

    async def test_resets_error_state_on_success(self):
        engine = _make_engine()
        engine._consecutive_errors = 3
        engine._error_backoff = 4.0

        async def fake_stream(*a, **kw):
            yield "ok"

        engine._llm.stream_response = fake_stream

        async for _ in engine.respond():
            pass

        assert engine._consecutive_errors == 0
        assert engine._error_backoff == 0.0


class TestToolLoop:
    async def test_executes_tool_and_yields_marker(self):
        engine = _make_engine()
        iteration = 0

        async def fake_stream(*a, **kw):
            nonlocal iteration
            if iteration == 0:
                yield "thinking..."
            else:
                yield "done"
            iteration += 1

        engine._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.name = "hass_request"
        tool_block.id = "tool_1"
        tool_block.input = {"command": "turn on lights"}

        engine._hass_agent = AsyncMock()
        engine._hass_agent.request = AsyncMock(return_value="OK")

        call_count = 0

        def tool_side_effect():
            nonlocal call_count
            call_count += 1
            return [tool_block] if call_count == 1 else []

        engine._llm.get_tool_use_blocks = tool_side_effect

        items = [item async for item in engine.respond()]

        text_chunks = [i for i in items if isinstance(i, str)]
        markers = [i for i in items if isinstance(i, ToolIterationMarker)]

        assert "done" in text_chunks
        assert len(markers) == 1
        engine._hass_agent.request.assert_awaited_once_with("turn on lights")

    async def test_tool_loop_exhaustion(self):
        engine = _make_engine()

        async def fake_stream(*a, **kw):
            yield "text"

        engine._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.name = "hass_request"
        tool_block.id = "tool_1"
        tool_block.input = {"command": "cmd"}

        engine._hass_agent = AsyncMock()
        engine._hass_agent.request = AsyncMock(return_value="OK")
        engine._llm.get_tool_use_blocks = MagicMock(return_value=[tool_block])

        async for _ in engine.respond():
            pass

        assert engine._hass_agent.request.await_count == 3


class TestCancel:
    async def test_cancel_interrupts_stream(self):
        engine = _make_engine()
        chunks = []

        async def slow_stream(*a, **kw):
            yield "first"
            await asyncio.sleep(0.1)
            yield "second"

        engine._llm.stream_response = slow_stream

        async for chunk in engine.respond():
            if isinstance(chunk, str):
                chunks.append(chunk)
                if chunk == "first":
                    engine.cancel()

        assert engine.last_result.interrupted is True
        assert "first" in chunks


class TestBusy:
    async def test_concurrent_respond_raises_busy(self):
        engine = _make_engine()

        async def slow_stream(*a, **kw):
            await asyncio.sleep(0.5)
            yield "slow"

        engine._llm.stream_response = slow_stream

        async def consume():
            async for _ in engine.respond():
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)

        with pytest.raises(BusyError):
            async for _ in engine.respond():
                pass

        engine.cancel()
        await task

    async def test_busy_cleared_after_respond(self):
        engine = _make_engine()

        async def fake_stream(*a, **kw):
            yield "hi"

        engine._llm.stream_response = fake_stream

        async for _ in engine.respond():
            pass

        assert engine.busy is False


class TestCompaction:
    async def test_compaction_triggers_memory_extraction(self):
        memory = AsyncMock()
        memory.extract_from_summary = AsyncMock(return_value={"semantic": []})
        memory.save_extraction = AsyncMock()

        engine = _make_engine(memory=memory)
        engine._llm.last_compaction_summary = "conversation summary"

        async def fake_stream(*a, **kw):
            yield "ok"

        engine._llm.stream_response = fake_stream

        async for _ in engine.respond():
            pass

        await asyncio.sleep(0.05)

        memory.extract_from_summary.assert_awaited_once_with("conversation summary")
        memory.save_extraction.assert_awaited_once()

    async def test_no_compaction_without_memory(self):
        engine = _make_engine()
        engine._llm.last_compaction_summary = "summary"

        async def fake_stream(*a, **kw):
            yield "ok"

        engine._llm.stream_response = fake_stream

        async for _ in engine.respond():
            pass

        assert engine.last_result.full_text == "ok"


class TestErrorBackoff:
    async def test_error_increments_backoff(self):
        engine = _make_engine()

        async def failing_stream(*a, **kw):
            raise RuntimeError("API error")
            yield  # make it a generator

        engine._llm.stream_response = failing_stream

        with pytest.raises(RuntimeError):
            async for _ in engine.respond():
                pass

        assert engine._consecutive_errors == 1
        assert engine._error_backoff > 0
        assert engine.busy is False


class TestAddUserMessage:
    def test_adds_to_context(self):
        engine = _make_engine()
        engine.add_user_message("hello")
        engine._ctx.add_message.assert_called_once_with("user", "hello")


class TestRAGLoading:
    def test_no_rag_task_without_memory(self):
        engine = _make_engine()
        engine.add_user_message("hello")
        assert engine._pending_rag is None

    async def test_rag_task_started_with_memory(self):
        memory = AsyncMock()
        memory.pre_load_context = AsyncMock(return_value="[semantic] fact")
        engine = _make_engine(memory=memory)
        engine.add_user_message("hello")
        assert engine._pending_rag is not None
        await engine._pending_rag
        engine._ctx.update_rag_context.assert_called_once_with("[semantic] fact")

    async def test_rag_applied_before_respond(self):
        memory = AsyncMock()
        memory.pre_load_context = AsyncMock(return_value="[semantic] fact")

        engine = _make_engine(memory=memory)

        async def fake_stream(*a, **kw):
            yield "ok"

        engine._llm.stream_response = fake_stream
        engine.add_user_message("hello")

        async for _ in engine.respond():
            pass

        engine._ctx.update_rag_context.assert_called()

    async def test_rag_failure_does_not_crash_respond(self):
        memory = AsyncMock()
        memory.pre_load_context = AsyncMock(side_effect=RuntimeError("embed failed"))

        engine = _make_engine(memory=memory)

        async def fake_stream(*a, **kw):
            yield "ok"

        engine._llm.stream_response = fake_stream
        engine.add_user_message("hello")

        async for _ in engine.respond():
            pass

        assert engine.last_result.full_text == "ok"
        assert engine.last_result.interrupted is False

    async def test_pending_rag_cleared_after_respond(self):
        memory = AsyncMock()
        memory.pre_load_context = AsyncMock(return_value="context")

        engine = _make_engine(memory=memory)

        async def fake_stream(*a, **kw):
            yield "ok"

        engine._llm.stream_response = fake_stream
        engine.add_user_message("hello")

        async for _ in engine.respond():
            pass

        assert engine._pending_rag is None


class TestShutdownSummarize:
    async def test_extracts_memories_on_shutdown(self):
        memory = AsyncMock()
        memory.generate_shutdown_summary = AsyncMock(return_value="summary")
        memory.extract_from_summary = AsyncMock(return_value={"semantic": []})
        memory.save_extraction = AsyncMock()

        engine = _make_engine(memory=memory)
        engine._ctx.get_messages.return_value = [{"role": "user", "content": "hi"}]

        await engine.shutdown_summarize()

        memory.generate_shutdown_summary.assert_awaited_once()
        memory.extract_from_summary.assert_awaited_once_with("summary")
        memory.save_extraction.assert_awaited_once()

    async def test_no_op_without_memory(self):
        engine = _make_engine()
        await engine.shutdown_summarize()

    async def test_no_op_without_messages(self):
        memory = AsyncMock()
        engine = _make_engine(memory=memory)
        engine._ctx.get_messages.return_value = []
        await engine.shutdown_summarize()
        memory.generate_shutdown_summary.assert_not_awaited()
