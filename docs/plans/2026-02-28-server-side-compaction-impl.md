# Server-side Compaction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use execute-flow to implement this plan task-by-task.

**Goal:** Replace client-side `TokenBudgetTrimmer` with Anthropic's server-side compaction + context editing (tool result clearing, thinking block clearing).

**Architecture:** Switch `LLMClient` from GA to Beta Messages API, adding `context_management.edits` with three strategies (thinking clearing, tool clearing, compaction). Remove `TokenBudgetTrimmer` entirely. Config migrates from `context_token_budget` to new compaction/clearing settings.

**Tech Stack:** Anthropic Python SDK (beta API), pytest, pytest-asyncio

---

### Task 1: Update config.py — swap old settings for new ones

**Files:**
- Modify: `src/prot/config.py:32-33`

**Step 1: Replace settings in config.py**

In `src/prot/config.py`, replace lines 32-33:

```python
    # OLD — delete these two lines:
    # context_token_budget: int = 30000
    # context_tool_result_max_chars: int = 2000

    # NEW — replace with:
    compaction_trigger: int = 50000
    tool_clear_trigger: int = 30000
    tool_clear_keep: int = 3
    thinking_keep_turns: int = 2
```

**Step 2: Run tests to verify**

Run: `uv run pytest tests/ -x -q`
Expected: No import errors. Tests referencing old settings won't fail (MagicMock accepts any attr) but will be cleaned up in Task 5.

**Step 3: Commit**

```bash
git add src/prot/config.py
git commit -m "feat: replace token budget config with compaction/clearing settings"
```

---

### Task 2: Migrate LLMClient to Beta API with context management

**Files:**
- Modify: `src/prot/llm.py` (surgical edits, NOT full replacement)
- Modify: `tests/test_llm.py`

**Step 1: Write the failing test**

In `tests/test_llm.py`, replace `test_stream_uses_ga_api` (lines 40-68) with:

```python
    async def test_stream_uses_beta_api_with_context_management(self):
        """stream_response uses beta messages.stream with compaction + context editing."""
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_stream.get_final_message = AsyncMock(
            return_value=MagicMock(content=[MagicMock(text="ok")])
        )

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response(
                system_blocks=[{"type": "text", "text": "test"}],
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass

            mock_client.beta.messages.stream.assert_called_once()
            call_kwargs = mock_client.beta.messages.stream.call_args.kwargs
            assert "compact-2026-01-12" in call_kwargs["betas"]
            assert "context-management-2025-06-27" in call_kwargs["betas"]
            assert "context_management" in call_kwargs
            edits = call_kwargs["context_management"]["edits"]
            edit_types = [e["type"] for e in edits]
            assert edit_types == [
                "clear_thinking_20251015",
                "clear_tool_uses_20250919",
                "compact_20260112",
            ]
            assert call_kwargs["thinking"] == {"type": "adaptive"}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py::TestLLMClient::test_stream_uses_beta_api_with_context_management -v`
Expected: FAIL — still calling GA API.

**Step 3: Apply surgical edits to llm.py**

Three changes in `src/prot/llm.py`:

**(a) Add module-level constants after `logger = get_logger(__name__)` (line 6):**

```python
_BETAS = ["compact-2026-01-12", "context-management-2025-06-27"]


def _build_context_management() -> dict:
    """Build context_management.edits from settings."""
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
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": settings.compaction_trigger},
            },
        ],
    }
```

**(b) In `stream_response`, change the API call (lines 31-43):**

```python
        # FROM:
        async with self._client.messages.stream(
            ...
            messages=messages,
            # --- Compaction (Opus 4.6 only, re-enable when Sonnet supports it) ---
            # betas=["compact-2026-01-12"],
            # context_management={
            #     "edits": [{"type": "compact_20260112"}],
            # },
        ) as stream:

        # TO:
        async with self._client.beta.messages.stream(
            ...
            messages=messages,
            betas=_BETAS,
            context_management=_build_context_management(),
        ) as stream:
```

Also update docstring (lines 23-26) to describe server-side context management.

**(c) In `count_tokens`, change the API call (lines 83-89):**

```python
        # FROM:
        result = await self._client.messages.count_tokens(
            model=settings.claude_model,
            system=system,
            tools=tools or [],
            messages=messages,
            thinking={"type": "adaptive"},
        )

        # TO:
        result = await self._client.beta.messages.count_tokens(
            model=settings.claude_model,
            system=system,
            tools=tools or [],
            messages=messages,
            thinking={"type": "adaptive"},
            betas=_BETAS,
            context_management=_build_context_management(),
        )
```

Also update docstring to note Beta API usage.

**Step 4: Update ALL existing tests in test_llm.py**

Global find-replace within `tests/test_llm.py`:

- `mock_client.messages.stream` → `mock_client.beta.messages.stream` (all occurrences)
- `mock_client.messages.count_tokens` → `mock_client.beta.messages.count_tokens` (all occurrences)
- `client._client.messages.stream` → `client._client.beta.messages.stream` (line 298, `test_stream_response_reset_content`)

**Step 5: Run all LLM tests**

Run: `uv run pytest tests/test_llm.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/prot/llm.py tests/test_llm.py
git commit -m "feat: migrate LLM client to beta API with server-side compaction + context editing"
```

---

### Task 3: Remove TokenBudgetTrimmer from pipeline

**Files:**
- Modify: `src/prot/pipeline.py`

**Step 1: Remove trimmer import and usage from pipeline.py**

Four deletions in `src/prot/pipeline.py`:

1. Delete import (line 11):
   ```python
   from prot.trimmer import TokenBudgetTrimmer
   ```

2. Delete trimmer instantiation in `_process_response` (lines 257-262):
   ```python
        trimmer = TokenBudgetTrimmer(
            llm=self._llm,
            system=system_blocks,
            tools=tools,
            budget=settings.context_token_budget,
        )
   ```

3. Remove `trimmer.fit()` call (line 267):
   ```python
   # FROM:
                messages = self._ctx.get_recent_messages()
                messages = await trimmer.fit(messages)
   # TO:
                messages = self._ctx.get_recent_messages()
   ```

4. Delete `trimmer.update_overhead()` call (line 407):
   ```python
                trimmer.update_overhead(self._llm.last_usage)
   ```

**Step 2: Run pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: ALL PASS.

**Step 3: Commit**

```bash
git add src/prot/pipeline.py
git commit -m "refactor: remove TokenBudgetTrimmer from pipeline (server-side compaction replaces it)"
```

---

### Task 4: Delete trimmer.py, test_trimmer.py, and logging dead code

**Files:**
- Delete: `src/prot/trimmer.py`
- Delete: `tests/test_trimmer.py`
- Modify: `src/prot/logging/constants.py:25`

**Step 1: Delete files**

```bash
rm src/prot/trimmer.py tests/test_trimmer.py
```

**Step 2: Remove trimmer entry from logging constants**

In `src/prot/logging/constants.py`, delete line 25:

```python
    "trimmer":          ("TRM", "\033[38;5;109m"),
```

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS.

**Step 4: Commit**

```bash
git add -u src/prot/trimmer.py tests/test_trimmer.py
git add src/prot/logging/constants.py
git commit -m "refactor: delete TokenBudgetTrimmer, its tests, and logging entry"
```

---

### Task 5: Clean up all stale references in pipeline tests

**Files:**
- Modify: `tests/test_pipeline.py`

**Step 1: Remove stale LLM mock attributes from `_make_pipeline()`**

Delete lines 42-43 (both `count_tokens` and `last_usage` mocks — pipeline no longer uses either after trimmer removal):

```python
    # DELETE — only used by removed TokenBudgetTrimmer:
    p._llm.count_tokens = AsyncMock(return_value=1000)
    p._llm.last_usage = MagicMock(input_tokens=1000)
```

**Step 2: Remove old config mock settings from 5 test methods**

Delete `context_token_budget` and `context_tool_result_max_chars` lines at these locations (pipeline no longer reads these settings):

- Lines 493-494 (inside `test_sentence_silence_delay`)
- Lines 536-537 (inside `test_no_silence_when_disabled`)
- Lines 1341-1342 (inside `test_circuit_breaker_*`)
- Lines 1376-1377 (inside `test_circuit_breaker_*`)
- Lines 1403-1404 (inside `test_backoff_caps_at_max`)

Each pair looks like:
```python
            ms.context_token_budget = 30000       # DELETE
            ms.context_tool_result_max_chars = 2000  # DELETE
```

**Step 3: Run pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: remove all stale trimmer/budget references from pipeline tests"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update Architecture section**

Remove the `trimmer.py` line:
```
  trimmer.py       # Token-budget context trimmer (count_tokens API + heuristic fallback)
```

**Step 2: Update Code Patterns section**

Replace the "Token-budget context" bullet:
```markdown
- **Token-budget context**: `TokenBudgetTrimmer` trims messages to fit within `context_token_budget` tokens.
  First call uses `count_tokens()` API; tool-loop iterations reuse `usage.input_tokens` from prior response.
  Long `tool_result` blocks are truncated to `context_tool_result_max_chars`. ContextManager stays a pure data
  container — trimming happens in pipeline via the trimmer.
```

With:
```markdown
- **Server-side compaction**: Beta API (`compact-2026-01-12` + `context-management-2025-06-27`)
  manages context automatically. Thinking clearing (keep 2 turns), tool result clearing (>30K tokens),
  and compaction (>50K tokens) all run server-side. No client-side trimming needed.
```

**Step 3: Update GA API bullet**

Replace:
```markdown
- **GA API**: Uses `messages.stream()` (not beta). Adaptive thinking + effort are GA on Sonnet 4.6.
```

With:
```markdown
- **Beta API**: Uses `beta.messages.stream()` with compaction + context editing betas.
  Adaptive thinking + effort are GA features passed through beta endpoint.
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for server-side compaction migration"
```

---

### Task 7: Full integration verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL PASS, zero import errors.

**Step 2: Verify no dead references**

Run: `grep -rn "trimmer\|TokenBudget\|context_token_budget\|context_tool_result_max_chars" src/ tests/ --include="*.py"`
Expected: Zero matches.

**Step 3: Verify the app starts (smoke test)**

Run: `uv run python -c "from prot.llm import LLMClient, _build_context_management; print(_build_context_management())"`
Expected: Prints the context_management dict with correct values from settings.
