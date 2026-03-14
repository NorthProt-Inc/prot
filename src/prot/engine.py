"""Interface-agnostic conversation engine — LLM streaming, tool loop, memory.

Shared core for both voice pipeline and text chat. No state machine,
no audio concerns. Callers consume the async generator from respond()
with their own concurrency model.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from prot.logging import get_logger, logged

logger = get_logger(__name__)

_MAX_TOOL_ITERATIONS = 3


class BusyError(Exception):
    """Raised when engine is already processing a response."""


@dataclass
class ResponseResult:
    full_text: str
    interrupted: bool


@dataclass
class ToolIterationMarker:
    """Yielded between tool loop iterations so callers can react."""
    iteration: int


class ConversationEngine:
    """Shared conversation core — LLM streaming, tool loop, memory.

    No state machine. No audio. Interface-agnostic.
    Callers consume the async generator from respond() with their own
    concurrency model (concurrent TTS for voice, direct send for text).
    """

    def __init__(
        self,
        ctx,
        llm,
        hass_agent=None,
        memory=None,
        graphrag=None,
    ):
        self._ctx = ctx
        self._llm = llm
        self._hass_agent = hass_agent
        self._memory = memory
        self._graphrag = graphrag
        self._busy = False
        self._response_parts: list[str] = []
        self._consecutive_errors = 0
        self._error_backoff = 0.0
        self._last_result: ResponseResult | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self._conversation_id = str(uuid.uuid4())
        self._pending_rag: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def last_result(self) -> ResponseResult | None:
        return self._last_result

    async def respond(self) -> AsyncGenerator[str | ToolIterationMarker, None]:
        """Stream LLM response as async generator.

        Yields str chunks and ToolIterationMarker between tool iterations.
        After generator exhausts, check self.last_result for outcome.

        Raises BusyError if already processing.
        """
        if self._busy:
            raise BusyError("Engine is already processing a response")
        self._busy = True
        self._llm._cancelled = False
        self._response_parts = []
        self._last_result = None
        try:
            async for item in self._respond_inner():
                yield item
        except Exception:
            self._consecutive_errors += 1
            self._error_backoff = min(
                5.0 * 2 ** max(0, self._consecutive_errors - 3), 30
            )
            logger.exception("respond() failed (backoff=%.1fs)", self._error_backoff)
            raise
        finally:
            self._busy = False

    async def _respond_inner(self) -> AsyncGenerator[str | ToolIterationMarker, None]:
        if self._error_backoff > 0:
            logger.info("Error backoff: sleeping %.1fs", self._error_backoff)
            await asyncio.sleep(self._error_backoff)

        if self._pending_rag is not None:
            await self._pending_rag
            self._pending_rag = None

        system_blocks = self._ctx.build_system_blocks()
        tools = self._ctx.build_tools(hass_agent=self._hass_agent)

        # +1 so the final iteration can produce a text response after
        # the last tool execution (tool calls are capped at _MAX_TOOL_ITERATIONS).
        for iteration in range(_MAX_TOOL_ITERATIONS + 1):
            messages = self._ctx.get_recent_messages()

            async for chunk in self._llm.stream_response(
                system_blocks, tools, messages
            ):
                if self._llm._cancelled:
                    break
                self._response_parts.append(chunk)
                yield chunk

            if self._llm._cancelled:
                self._last_result = ResponseResult(
                    full_text="".join(self._response_parts), interrupted=True
                )
                return

            tool_blocks = self._llm.get_tool_use_blocks()

            if not tool_blocks:
                full_text = self._commit_response()
                self._handle_compaction()
                self._consecutive_errors = 0
                self._error_backoff = 0.0
                self._last_result = ResponseResult(
                    full_text=full_text, interrupted=False
                )
                return

            if iteration >= _MAX_TOOL_ITERATIONS:
                # All tool iterations spent — commit whatever text we have
                logger.error("Tool loop exhausted after %d iterations", _MAX_TOOL_ITERATIONS)
                full_text = self._commit_response()
                self._last_result = ResponseResult(
                    full_text=full_text, interrupted=False
                )
                return

            # Tool use — commit partial response, signal callers, execute
            self._commit_response()
            yield ToolIterationMarker(iteration=iteration)
            tool_results = await self._execute_tools(tool_blocks)
            self._ctx.add_message("user", tool_results)

            if self._llm._cancelled:
                self._last_result = ResponseResult(
                    full_text="".join(self._response_parts), interrupted=True
                )
                return

            self._response_parts = []

    def _commit_response(self) -> str:
        full_text = "".join(self._response_parts)
        content = self._llm.last_response_content
        self._ctx.add_message("assistant", content or full_text)
        self._save_message_bg("assistant", full_text)
        return full_text

    async def _execute_tools(self, tool_blocks) -> list:
        results = []
        for block in tool_blocks:
            try:
                if block.name == "hass_request" and self._hass_agent:
                    result = await self._hass_agent.request(block.input["command"])
                else:
                    logger.warning("Unknown tool: %s", block.name)
                    result = f"Unknown tool: {block.name}"
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )
            except Exception as exc:
                logger.warning("Tool %s failed: %s", block.name, exc)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {exc}",
                        "is_error": True,
                    }
                )
        return results

    def _handle_compaction(self) -> None:
        if not self._memory:
            return
        summary = self._llm.last_compaction_summary
        if not summary:
            return
        self._bg(self._process_compaction(summary))

    async def _preload_rag(self, query: str) -> None:
        """Background RAG context loading — errors are swallowed."""
        try:
            rag = await self._memory.pre_load_context(query)
            self._ctx.update_rag_context(rag)
        except Exception:
            logger.warning("RAG preload failed", exc_info=True)

    async def _process_compaction(self, summary: str) -> None:
        try:
            extraction = await self._memory.extract_from_summary(summary)
            await self._memory.save_extraction(extraction)
        except Exception:
            logger.warning("Compaction memory extraction failed", exc_info=True)

    def add_user_message(self, text: str) -> None:
        self._ctx.add_message("user", text)
        self._save_message_bg("user", text)
        if self._memory:
            self._pending_rag = self._bg(self._preload_rag(text))

    def cancel(self) -> None:
        self._llm.cancel()

    async def shutdown_summarize(self) -> None:
        if not self._memory:
            return
        messages = self._ctx.get_messages()
        if not messages:
            return
        try:
            summary = await self._memory.generate_shutdown_summary(messages)
            extraction = await self._memory.extract_from_summary(summary)
            await self._memory.save_extraction(extraction)
            logger.info("Shutdown memory extraction complete")
        except Exception:
            logger.warning("Shutdown summarization failed", exc_info=True)

    def _save_message_bg(self, role: str, content: str) -> None:
        if not self._graphrag:
            return

        async def _save():
            try:
                await self._graphrag.save_message(
                    self._conversation_id, role, content
                )
            except Exception:
                logger.warning("Failed to save message to DB", exc_info=True)

        self._bg(_save())

    def _bg(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task
