# Time Awareness + RAG Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use execute-flow to implement this plan task-by-task.

**Goal:** Fix two bugs — (1) time awareness failure after compaction, (2) RAG retrieval dead code never called.

**Architecture:** Three-layer approach for time awareness: compaction instructions to preserve temporal markers in summaries, session timeline in system prompt Block 3 (survives compaction), and message timestamp tracking. For RAG: eager background loading on user message arrival, awaited before LLM call.

**Tech Stack:** Python / Anthropic SDK beta API / asyncio / pytest

---

### Task 1: Compaction Instructions for Temporal Preservation

**Files:**
- Modify: `src/prot/llm.py:11-33` (`_build_context_management`)
- Test: `tests/test_llm.py`

**Step 1: Write the failing test**

Add to `tests/test_llm.py` inside `TestCompactionDetection`:

```python
async def test_compaction_edit_includes_instructions(self):
    """Compaction edit should include instructions for temporal preservation."""
    with patch("prot.llm.settings") as ms:
        ms.thinking_keep_turns = 2
        ms.tool_clear_trigger = 30000
        ms.tool_clear_keep = 3
        ms.compaction_trigger = 50000
        ms.pause_after_compaction = True

        from prot.llm import _build_context_management
        cm = _build_context_management()
        compact_edit = cm["edits"][2]
        assert "instructions" in compact_edit
        assert isinstance(compact_edit["instructions"], str)
        assert len(compact_edit["instructions"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py::TestCompactionDetection::test_compaction_edit_includes_instructions -v`
Expected: FAIL — `"instructions" not in compact_edit`

**Step 3: Write minimal implementation**

In `src/prot/llm.py`, modify `_build_context_management()`. Add `instructions` to `compact_edit`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py::TestCompactionDetection -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/prot/llm.py tests/test_llm.py
git commit -m "feat: add compaction instructions for temporal marker preservation"
```

---

### Task 2: Block 3 Session Timeline

**Files:**
- Modify: `src/prot/context.py:1-105`
- Test: `tests/test_context.py`

**Step 1: Write failing tests**

Add to `tests/test_context.py`:

```python
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo

_TEST_TZ = ZoneInfo("America/Vancouver")


class TestBlock3Timeline:
    def test_block3_contains_session_start_after_messages(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "hello")
        blocks = cm.build_system_blocks()
        assert "session_start:" in blocks[2]["text"]

    def test_block3_contains_recent_turns(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "hello")
        cm.add_message("assistant", "hi")
        blocks = cm.build_system_blocks()
        text = blocks[2]["text"]
        assert "recent_turns:" in text
        assert "user" in text
        assert "assistant" in text

    def test_block3_no_timeline_without_messages(self):
        cm = ContextManager(persona_text="test", rag_context="")
        blocks = cm.build_system_blocks()
        assert "session_start:" not in blocks[2]["text"]
        assert "recent_turns:" not in blocks[2]["text"]

    def test_message_times_excludes_tool_results(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "hello")
        cm.add_message("assistant", "thinking...")
        cm.add_message("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}])
        cm.add_message("assistant", "done")
        # tool_result should not appear in timeline
        assert len(cm._message_times) == 3  # user, assistant, assistant (not tool_result)

    def test_recent_turns_limited_to_last_10(self):
        cm = ContextManager(persona_text="test", rag_context="")
        for i in range(20):
            cm.add_message("user", f"msg-{i}")
            cm.add_message("assistant", f"resp-{i}")
        blocks = cm.build_system_blocks()
        lines = [l for l in blocks[2]["text"].split("\n") if l.strip().startswith("- ")]
        assert len(lines) == 10
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context.py::TestBlock3Timeline -v`
Expected: FAIL — `_message_times` attribute not found

**Step 3: Write minimal implementation**

Modify `src/prot/context.py`:

```python
from datetime import datetime

from prot.config import LOCAL_TZ
from prot.processing import is_tool_result_message


class ContextManager:
    """Build and manage the multi-block system prompt for Claude API calls.

    The 3-block layout is designed for optimal prompt caching:
      Block 1: Persona + Rules      (STATIC, cached)
      Block 2: GraphRAG Context      (TOPIC-DEPENDENT, cached)
      Block 3: Dynamic Context       (PER-REQUEST, NOT cached, MUST be last)

    Placing dynamic content last preserves the cached prefix so that
    blocks 1 and 2 can hit the prompt cache on subsequent requests.
    """

    def __init__(self, persona_text: str, rag_context: str = "", channel: str = "voice") -> None:
        self._persona = persona_text
        self._rag_context = rag_context
        self._channel = channel
        self._messages: list[dict] = []
        self._session_start: datetime = datetime.now(LOCAL_TZ)
        self._message_times: list[tuple[str, datetime]] = []

    def build_system_blocks(self) -> list[dict]:
        """Build 3-block system prompt with cache control.

        Order is CRITICAL for prompt caching:
          Block 1: Persona + Rules (STATIC, cached)
          Block 2: GraphRAG Context (TOPIC-DEPENDENT, cached)
          Block 3: Dynamic Context (PER-REQUEST, NOT cached, MUST be last)

        If dynamic content (datetime) sits between cached blocks, it breaks the
        cache prefix -- downstream blocks would NEVER hit cache.
        """
        block1_persona: dict = {
            "type": "text",
            "text": self._persona,
            "cache_control": {"type": "ephemeral"},
        }
        block2_rag: dict = {
            "type": "text",
            "text": self._rag_context or "(no additional context)",
            "cache_control": {"type": "ephemeral"},
        }
        now = datetime.now(LOCAL_TZ)
        block3_text = (
            f"datetime: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"timezone: America/Vancouver\n"
            f"channel: {self._channel}"
        )
        timeline = self._build_timeline()
        if timeline:
            block3_text += f"\n{timeline}"
        block3_dynamic: dict = {
            "type": "text",
            "text": block3_text,
        }
        return [block1_persona, block2_rag, block3_dynamic]

    def _build_timeline(self) -> str:
        """Build session timeline for Block 3 temporal context."""
        if not self._message_times:
            return ""
        lines = [f"session_start: {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}"]
        recent = self._message_times[-10:]
        lines.append("recent_turns:")
        for role, ts in recent:
            lines.append(f"  - {ts.strftime('%H:%M')} {role}")
        return "\n".join(lines)

    def build_tools(self, hass_agent=None) -> list[dict]:
        """Build tool definitions with cache on last tool."""
        web_search: dict = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 1,
            "user_location": {
                "type": "approximate",
                "city": "Vancouver",
                "country": "CA",
                "timezone": "America/Vancouver",
            },
        }
        tools = [web_search]

        if hass_agent is not None:
            tools.append(hass_agent.build_tool())

        # Ensure last tool has cache_control
        if "cache_control" not in tools[-1]:
            tools[-1]["cache_control"] = {"type": "ephemeral"}

        return tools

    def add_message(self, role: str, content: str | list) -> None:
        """Append a message. Content can be str or list of content blocks."""
        msg = {"role": role, "content": content}
        self._messages.append(msg)
        if not is_tool_result_message(msg):
            self._message_times.append((role, datetime.now(LOCAL_TZ)))

    def get_messages(self) -> list[dict]:
        """Return a copy of the conversation history."""
        return list(self._messages)

    def get_recent_messages(self) -> list[dict]:
        """Return conversation history with orphan boundary correction.

        Ensures the window starts at a real user message
        (not an orphaned tool_result or assistant message).
        """
        messages = list(self._messages)
        while messages and (
            messages[0]["role"] != "user"
            or is_tool_result_message(messages[0])
        ):
            messages = messages[1:]
        return messages

    def update_rag_context(self, context: str) -> None:
        """Replace the RAG context for the next system prompt build."""
        self._rag_context = context
```

**Step 4: Run all context tests**

Run: `uv run pytest tests/test_context.py -v`
Expected: ALL PASS (new + existing)

**Step 5: Commit**

```bash
git add src/prot/context.py tests/test_context.py
git commit -m "feat: add session timeline to Block 3 for temporal awareness"
```

---

### Task 3: Per-Turn RAG Loading

**Files:**
- Modify: `src/prot/engine.py:1-249`
- Test: `tests/test_engine.py`

**Step 1: Write failing tests**

Add to `tests/test_engine.py`:

```python
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
        memory.extract_from_summary = AsyncMock(return_value={"semantic": []})
        memory.save_extraction = AsyncMock()

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
        memory.extract_from_summary = AsyncMock(return_value={"semantic": []})
        memory.save_extraction = AsyncMock()

        engine = _make_engine(memory=memory)

        async def fake_stream(*a, **kw):
            yield "ok"

        engine._llm.stream_response = fake_stream
        engine.add_user_message("hello")

        async for _ in engine.respond():
            pass

        # Response should still succeed despite RAG failure
        assert engine.last_result.full_text == "ok"
        assert engine.last_result.interrupted is False

    async def test_pending_rag_cleared_after_respond(self):
        memory = AsyncMock()
        memory.pre_load_context = AsyncMock(return_value="context")
        memory.extract_from_summary = AsyncMock(return_value={"semantic": []})
        memory.save_extraction = AsyncMock()

        engine = _make_engine(memory=memory)

        async def fake_stream(*a, **kw):
            yield "ok"

        engine._llm.stream_response = fake_stream
        engine.add_user_message("hello")

        async for _ in engine.respond():
            pass

        assert engine._pending_rag is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine.py::TestRAGLoading -v`
Expected: FAIL — `_pending_rag` attribute not found

**Step 3: Write minimal implementation**

Modify `src/prot/engine.py`:

Add `_pending_rag` to `__init__`:

```python
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
```

Add `_preload_rag` method (after `_handle_compaction`):

```python
async def _preload_rag(self, query: str) -> None:
    """Background RAG context loading — errors are swallowed."""
    try:
        rag = await self._memory.pre_load_context(query)
        self._ctx.update_rag_context(rag)
    except Exception:
        logger.warning("RAG preload failed", exc_info=True)
```

Modify `add_user_message`:

```python
def add_user_message(self, text: str) -> None:
    self._ctx.add_message("user", text)
    self._save_message_bg("user", text)
    if self._memory:
        self._pending_rag = self._bg(self._preload_rag(text))
```

Modify `_respond_inner` — add RAG await before `build_system_blocks()`:

```python
async def _respond_inner(self) -> AsyncGenerator[str | ToolIterationMarker, None]:
    if self._error_backoff > 0:
        logger.info("Error backoff: sleeping %.1fs", self._error_backoff)
        await asyncio.sleep(self._error_backoff)

    if self._pending_rag is not None:
        await self._pending_rag
        self._pending_rag = None

    system_blocks = self._ctx.build_system_blocks()
    tools = self._ctx.build_tools(hass_agent=self._hass_agent)
    # ... rest unchanged
```

**Step 4: Run all engine tests**

Run: `uv run pytest tests/test_engine.py -v`
Expected: ALL PASS (new + existing)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/prot/engine.py tests/test_engine.py
git commit -m "feat: wire RAG retrieval into conversation engine"
```

---

### Task 4: Integration Verification

**Step 1: Verify all tests pass together**

Run: `uv run pytest --tb=short -q`
Expected: All tests pass, 0 failures

**Step 2: Verify existing tests unchanged behavior**

Run: `uv run pytest tests/test_context.py tests/test_engine.py tests/test_llm.py -v`
Expected: All existing + new tests pass

**Step 3: Final commit (if any fixups needed)**

```bash
git add -u
git commit -m "fix: test adjustments for time awareness + RAG integration"
```
