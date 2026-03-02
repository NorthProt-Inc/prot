# Simplification Plan — 2026-03-01

Post server-side compaction migration audit. 5-perspective analysis (Complexity, DRY, YAGNI, Dead Code, Coupling).

## Summary

- Files analyzed: 28 source, 30 test
- Total findings: 21 (after false positive screening from 40+ raw)
- Estimated lines removable: ~80 Python + ~15 SQL
- Worker results: CX: 6 findings, DRY: 8, YG: 5, DC: 3, CP: 3

## Task 1: Dead code & YAGNI cleanup

**Findings**: YG-1+DC-1, DC-3, YG-3 | Score: 3.5 | Risk: NONE

### 1a. Delete deprecated config field

**File**: `src/prot/config.py:65`

Delete:
```python
memory_extraction_window_turns: int = 3  # deprecated: kept for config compat
```

Zero references in code or `.env`. The field and its comment are unused dead code.

### 1b. Delete dead conftest fixtures

**File**: `tests/conftest.py:12-19`

Delete `sample_transcript` and `sample_llm_response` fixtures. Zero test references.

### 1c. Remove unused `_trace_dir` metadata

**Files**:
- `src/prot/logging/tracing.py` — remove `"_trace_dir": "..."` from 4 dict literals (lines 96, 112, 117, 130)
- `src/prot/logging/formatters.py` — remove `extra_data.pop("_trace_dir", None)` (line 24)

Written 4 times, popped and discarded every time. Never consumed.

**Verify**: `uv run pytest tests/test_logging_tracing.py tests/test_logging_formatters.py -v`

---

## Task 2: Inline execute_tool stub

**Findings**: YG-2+DC-4 | Score: 3.5 | Risk: LOW

The `LLMClient.execute_tool()` method always returns `{"error": ...}`. Only reached when an unknown tool name appears, which is impossible with current tool config (web_search is server-side, hass_request has its own branch).

### 2a. Inline the error in pipeline.py

**File**: `src/prot/pipeline.py:380-383`

Before:
```python
else:
    result = await self._llm.execute_tool(block.name, block.input)
```

After:
```python
else:
    logger.warning("Unknown tool", tool=block.name)
    result = {"error": f"Unknown tool: {block.name}"}
```

### 2b. Delete execute_tool from LLMClient

**File**: `src/prot/llm.py:107-110` — delete the `@logged` decorator and method

### 2c. Delete execute_tool test

**File**: `tests/test_llm.py` — delete `test_execute_tool_unknown_returns_error`

**Verify**: `uv run pytest tests/test_llm.py tests/test_pipeline.py -v`

---

## Task 3: Extract `strip_markdown_fences` utility

**Findings**: DRY-4 | Score: 3.8 | Risk: NONE

Exact duplicate in `memory.py:143-144` and `community.py:204-205`.

### 3a. Add to processing.py

**File**: `src/prot/processing.py`

```python
def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses."""
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text
```

### 3b. Replace in memory.py and community.py

Both files: replace inline fence stripping with `strip_markdown_fences(raw)`.
Add `strip_markdown_fences` to import from `prot.processing`.

**Verify**: `uv run pytest tests/test_memory.py tests/test_community.py -v`

---

## Task 4: Simplify community.py

**Findings**: CX-5+DRY-6 | Score: 3.6 | Risk: LOW

### 4a. Extract `_clear_communities()` method

**File**: `src/prot/community.py`

The 3-line cleanup block (`rebuild_communities([])` + log + `return 0`) appears 3 times (lines 89-91, 109-111, 135-136). Extract:

```python
async def _clear_communities(self) -> int:
    await self._store.rebuild_communities([])
    logger.info("No communities detected, cleared stale data")
    return 0
```

Replace all 3 occurrences with `return await self._clear_communities()`.

**Verify**: `uv run pytest tests/test_community.py -v`

---

## Task 5: Extract `_commit_assistant_response` in pipeline.py

**Findings**: DRY-8 | Score: 3.2 | Risk: LOW

4-line block duplicated between normal completion (line ~354) and tool-use path (line ~372).

**File**: `src/prot/pipeline.py`

```python
def _commit_assistant_response(self, parts: list[str]) -> str:
    full_text = "".join(parts)
    content = self._llm.last_response_content
    self._ctx.add_message("assistant", content or full_text)
    self._save_message_bg("assistant", full_text)
    return full_text
```

Replace both inline blocks with `full_text = self._commit_assistant_response(_response_parts)`.

**Verify**: `uv run pytest tests/test_pipeline.py -v`

---

## Task 6: Fix formatter mutation bug

**Findings**: CX-9 | Score: 3.4 | Risk: NONE

**File**: `src/prot/logging/formatters.py`

`_prepare_record()` mutates `extra_data` via `pop()`. When both SmartFormatter (console)
and PlainFormatter (file) process the same record, the second formatter sees `_depth`,
`_trace_dir`, `_elapsed` already removed — trace metadata is silently lost in the file log.

Replace with non-mutating reads:

```python
# Before (actual code, lines 22-27)
trace_depth = extra_data.pop("_depth", None)
extra_data.pop("_trace_dir", None)
trace_elapsed = extra_data.pop("_elapsed", None)

kv_parts = [f"{k}={v}" for k, v in extra_data.items()]

# After
trace_depth = extra_data.get("_depth", None)
trace_elapsed = extra_data.get("_elapsed", None)

kv_parts = [f"{k}={v}" for k, v in extra_data.items() if not k.startswith("_")]
```

### 6b. Add mutation regression test

**File**: `tests/test_logging_formatters.py`

Add a test that calls both SmartFormatter and PlainFormatter on the same record
and verifies the second formatter still sees `_depth` and `_elapsed`.

**Verify**: `uv run pytest tests/test_logging_formatters.py -v`

---

## Task 7: Schema dead columns cleanup

**Findings**: YG-4, YG-6, YG-7 | Score: 3.0 | Risk: MEDIUM (requires DB migration)

**File**: `src/prot/schema.sql`

### Removals (dead columns only — pg_trgm kept for scripts/db_gc.py):
1. `attributes JSONB` from `entities` — never read or written
2. `attributes JSONB` from `relationships` — never read or written
3. `content_embedding vector(1024)` from `conversation_messages` — never populated
4. `idx_messages_content_embedding` — index on never-populated column
5. `metadata JSONB` from `conversation_messages` — never read or written

### Kept (has dependents):
- `pg_trgm` extension — used by `scripts/db_gc.py` `similarity()` function
- `idx_entities_name_trgm` — supports `db_gc.py` trigram queries

### Also:
- `src/prot/graphrag.py`: remove `embedding` parameter from `save_message()`

**Verify**: `uv run pytest tests/test_graphrag.py -v`

**NOTE**: Existing databases need a migration. For dev, re-create. For prod (if any), write ALTER TABLE statements.

---

## Task 8: Voyage client DRY

**Findings**: DRY-1+DRY-2 | Score: 2.9 | Risk: LOW

Both `embeddings.py` and `reranker.py` have identical:
- Client construction: `voyageai.AsyncClient(api_key=api_key or settings.voyage_api_key)`
- Close method: duck-typed `close()/aclose()` fallback

Extract shared close utility into `embeddings.py` (or a new `_voyage.py`):

```python
async def _close_voyage_client(client) -> None:
    if hasattr(client, "close"):
        await client.close()
    elif hasattr(client, "aclose"):
        await client.aclose()
```

Both classes call this in their `close()` methods.

**Verify**: `uv run pytest tests/test_embeddings.py tests/test_reranker.py -v`

---

## Task 9: Add Pipeline.current_state property

**Findings**: CP-5 | Score: 3.4 | Risk: NONE

**File**: `src/prot/pipeline.py`

```python
@property
def current_state(self) -> str:
    return self._sm.state.value
```

**File**: `src/prot/app.py` — replace `pipeline.state.state.value` with `pipeline.current_state`

**File**: `tests/test_app.py` — update mock setup: replace `mock_pipeline.state.state.value = "..."` with `mock_pipeline.current_state = "..."` in both test methods (lines 13, 38).

**Verify**: `uv run pytest tests/test_app.py -v`

---

## Deferred (out of scope)

| Finding | Reason |
|---------|--------|
| CX-1+CX-2: pipeline._process_response 194 lines | High risk, needs careful extraction of producer-consumer + tool loop |
| DRY-3+CP-2: 3x AsyncAnthropic clients | Needs lifecycle management design |
| CP-1: Settings singleton DI | Massive scope — 11 modules |
| CP-3: db._pool global | Low impact, needs careful state management |
| CP-8: Logging deferred import | Low impact |

## Execution Order

Tasks are independent except:
- Task 3 (strip_markdown_fences) should precede Task 4 (community.py imports processing)

Recommended parallel groups:
- Group A (independent): 1, 2, 6, 9
- Group B (after A): 3, 8
- Group C (after B): 4, 5, 7
