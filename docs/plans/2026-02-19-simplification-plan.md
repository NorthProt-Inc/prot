# Simplification Plan: prot

**Date:** 2026-02-19
**Files analyzed:** 30 source files (~3,646 lines)
**Raw findings:** 57 (across 5 workers: CX:10, DRY:10, YG:17, DC:10, CP:10)
**After screening:** 38 findings merged into 9 refactoring units
**Estimated total lines removable:** ~270
**Review status:** PASS WITH CHANGES (applied)

---

## Discarded Findings

- **CX-9** (MemoryExtractor 5-param constructor): DISCARDED — explicit dependency injection is a good pattern
- **YG-13** (playback.py input validation): DISCARDED — defensive validation in constructors is standard practice
- **YG-16** (per-module color duplicates in constants): DISCARDED — cosmetic, no maintainability impact
- **YG-17** (tracemalloc /memory endpoint): DISCARDED — debug tooling is acceptable overhead (15 lines, opt-in)
- **DC-8** (db.py module-level _pool): DISCARDED — double-init guard is useful defensive code
- **DC-9** (logging __init__ exports): DISCARDED — valid public API even if only test-consumed
- **CP-7** (content_to_text placement): DISCARDED — module placement is style, not simplification
- **CP-10** (runtime imports in startup): DISCARDED — intentional for graceful degradation of optional deps
- **CP-2** (Share Anthropic clients): DISCARDED (review) — introduces coupling for ~6 lines saved; three independent clients is simpler and more robust
- **CP-3/CP-5** (Voyage client consolidation + "reranker leak"): DISCARDED (review) — reranker IS already closed via `MemoryExtractor.close()` (no leak); client sharing adds coupling
- **CX-1/CP-9** (_process_response refactoring): DISCARDED (review) — HIGH risk to critical real-time path for subjective readability gains; should be a separate carefully-reviewed PR

---

## Refactoring Units (Priority Order)

### Task 1: Simplify logging/tracing.py [Score: 4.4]
**Perspectives:** CX-2 + CX-3 + DRY-1 + YG-1 + YG-2 + YG-3 + YG-4 | Severity: HIGH
**Impact:** ~130 lines removed from 330-line file

The tracing module has significant dead code and duplication:
1. `_trace_sync` (58 lines) is **never called** — all 10 `@logged()` sites decorate async functions
2. `_SECRET_RE` + redaction logic (~20 lines) — no function args match secret patterns
3. `log_result` parameter + `_fmt_result` — never set to True by any caller
4. `max_val_len` parameter — never overridden from default 80
5. `_trace_async` and `_trace_async_gen` share ~75% identical logic

**Steps:**
1. Remove `_trace_sync` function entirely (lines 109-166)
2. Remove `_SECRET_RE`, `redact` parameter, and redaction conditionals from `_fmt_args`
3. Remove `log_result` parameter, `_fmt_result` function, and result-logging branches
4. Hardcode `max_val_len=80` in `_fmt_val`, remove parameter from decorator/wrappers
5. Extract shared logic from `_trace_async` and `_trace_async_gen` into helpers: `_log_entry()`, `_log_exit()`
   **Note:** `_trace_async_gen` does not support log_result (generators can't capture a return value). The shared helpers must account for this asymmetry.
6. Update `logged()` dispatcher to remove sync branch

**Verification:** `uv run pytest tests/test_logging_tracing.py -v` — update tests that test removed features

---

### Task 2: Clean up embeddings.py [Score: 4.2]
**Perspectives:** DC-2 + YG-9 + YG-10 + YG-11 | Severity: MEDIUM
**Impact:** ~30 lines removed

1. Remove `embed_texts()` — zero production callers (only test refs)
2. Remove `embed_query()` — zero production callers
3. Remove `MAX_BATCH` constant (only used by deleted `embed_texts`)
4. Remove `max_concurrent` param and `_semaphore` — removing a defensive concurrency guard for an external API; currently no concurrent embedding calls exist. Acknowledge this is removing a safety guard, not just dead code.

**Steps:**
1. Delete `embed_texts` method (lines 33-45)
2. Delete `embed_query` method (lines 47-55)
3. Delete `MAX_BATCH = 128` constant (line 12)
4. Remove `max_concurrent` param from `__init__`, delete `_semaphore` field
5. Remove `async with self._semaphore:` wrapping from remaining methods
6. Update tests in `test_embeddings.py` to remove tests for deleted methods

**Verification:** `uv run pytest tests/test_embeddings.py tests/test_memory.py tests/test_community.py -v`

---

### Task 3: Deduplicate logging/formatters.py [Score: 3.6]
**Perspectives:** CX-4 + DRY-2 + YG-5 | Severity: MEDIUM
**Impact:** ~10 lines removed (corrected from initial estimate of ~25)

`SmartFormatter.format()` and `PlainFormatter.format()` share ~14 lines of identical logic:
- trace metadata extraction (`_depth`, `_trace_dir`, `_elapsed` popping)
- k=v parts building with elapsed time formatting
- indent computation
- exc_info/exc_text handling

**Steps:**
1. Extract `_prepare_record(record) -> tuple[list[str], str]` that returns `(kv_parts, indent)` and mutates record's extra_data
2. `SmartFormatter.format()` calls `_prepare_record()`, then does colored line assembly
3. `PlainFormatter.format()` calls `_prepare_record()`, then does plain line assembly

**Verification:** `uv run pytest tests/test_logging_formatters.py -v`

---

### Task 4: Remove log.py facade and migrate imports [Score: 3.4]
**Perspectives:** CP-1 + DC-6 + YG-15 | Severity: MEDIUM
**Impact:** ~10 lines removed, cleaner import path

All 12 production files import from `prot.log` (a re-export facade). Migrate to canonical `prot.logging` path.

**Steps:**
1. Migrate all 12 files from `from prot.log import ...` to `from prot.logging import ...`:
   - `pipeline.py`, `app.py`, `llm.py`, `memory.py`, `community.py`, `tts.py`
   - `stt.py`, `db.py`, `hass.py`, `playback.py`, `audio.py`, `conversation_log.py`
2. Delete `log.py`
3. Update any test files that import from `prot.log`

**Note (from review):** Steps 3-6 from the original plan are DROPPED:
- `StructuredLogger`, `SmartFormatter`, `PlainFormatter` exports are used by test files — NOT unused
- Inlining `handlers.py` and `constants.py` is just moving code with no simplification gain
- Removing error-only log file is a functionality change, not dead code

**Verification:** `uv run pytest -v` (full test suite — touches many import paths)

---

### Task 5: GraphRAG conn-or-pool helper [Score: 3.2]
**Perspectives:** CX-8 + DRY-3 + DC-3 + YG-8 | Severity: MEDIUM
**Impact:** ~18 lines removed

Three methods repeat `if conn: use conn, else: acquire from pool` pattern.

**Steps:**
1. Add async context manager `_conn(self, conn=None)` to `GraphRAGStore`
2. Simplify `upsert_entity`, `upsert_relationship`, `get_entity_id_by_name` to use it
3. Remove `max_depth` parameter from `get_entity_neighbors` (documented as unused, always passed as 1)
4. Update caller in `memory.py:209` to remove `max_depth=1` kwarg

**Verification:** `uv run pytest tests/test_graphrag.py tests/test_memory.py -v`

---

### Task 6: Fix FSM encapsulation [Score: 2.9]
**Perspectives:** CP-4 + DC-1 | Severity: HIGH (design quality)
**Impact:** architectural improvement

Pipeline directly writes `self._sm._state` in 4 error-recovery paths, bypassing FSM validation.

**Steps:**
1. Add `StateMachine.force_recovery(target: State)` method to `state.py` — should log a warning to preserve debuggability
2. Replace all 4 `self._sm._state = State.XXX` in `pipeline.py` with `self._sm.force_recovery(State.XXX)`
   - Line 189: STT connect failure → force IDLE
   - Line 381: Tool loop exhaustion → force ACTIVE
   - Line 393: Exception recovery → force ACTIVE
   - Line 413: STT reconnect failure → force IDLE
3. Remove unused `on_tts_complete()` from `StateMachine` (only `try_on_tts_complete` is used in production)
4. Update test files that use `on_tts_complete()`:
   - `test_state.py`: lines 33, 41, 67, 119 — switch to `try_on_tts_complete()` or direct state setup
   - `test_pipeline.py`: lines 116, 371 — switch to `try_on_tts_complete()`

**Verification:** `uv run pytest tests/test_state.py tests/test_pipeline.py -v`

---

### Task 7: Pipeline STT reconnect dedup [Score: 2.6]
**Perspectives:** DRY-6 | Severity: MEDIUM
**Impact:** ~13 lines removed

`_handle_vad_speech()` and `_handle_barge_in()` share identical STT reconnect sequence (~10 lines).

**Steps:**
1. Extract `async def _reconnect_stt(self) -> bool` with shared logic:
   drain prebuffer, reset VAD, connect STT, check failure, flush pending, set connected
2. Simplify both callers to use helper
3. Each caller handles its own pre/post (start_turn, state transitions, cancellation)

**Verification:** `uv run pytest tests/test_pipeline.py -v`

---

### Task 8: Small dead code & YAGNI cleanups [Score: various LOW]
**Perspectives:** DC-4 + DC-5 + DC-7 + DC-10 + YG-7 + YG-12 | Severity: LOW
**Impact:** ~15 lines removed

**Steps:**
1. Remove `LLMClient._active_stream` field and 3 assignments (DC-4, written but never read)
2. Implement `LLMClient.close()` to actually close the Anthropic client (DC-5, currently empty no-op)
3. Remove `StructuredLogger.name` property (DC-7, never accessed externally; `_log()` uses `self._logger.name` directly)
4. Remove `StructuredLogger.critical()` method (YG-7, zero callers in production)
5. Keep `execute_tool` in llm.py as a defensive stub but rename to clarify it's a fallback (DC-10). Do NOT inline into pipeline.py — that worsens separation of concerns.
6. Remove `voyage_dimension` from config.py (YG-12, never referenced outside config)

**Verification:** `uv run pytest -v`

---

### Task 9: Remaining DRY cleanups [Score: various LOW]
**Perspectives:** DRY-5 + DRY-7 + DRY-8 + DRY-9 + DRY-10 | Severity: LOW
**Impact:** ~30 lines removed

**Steps:**
1. Extract `_rerank_if_available()` helper in memory.py (DRY-5, ~8 lines saved)
2. Extract `_bg(self, coro) -> Task` helper in pipeline.py for background task fire-and-forget (DRY-7, ~6 lines saved)
3. Define `LOCAL_TZ` in config.py, import in context.py and conversation_log.py (DRY-8, ~2 lines + consistency)
4. Cache `_known_ids` set in HassRegistry.discover() (DRY-9, ~6 lines saved)
5. Extract `async def _kill_process(self)` in playback.py (DRY-10, ~10 lines saved)

**Verification:** `uv run pytest -v`

---

## Execution Order & Dependencies

```
Task 1 (tracing.py) ────┐
Task 2 (embeddings.py) ──┤
Task 3 (formatters.py) ──┼── independent, can run in parallel
Task 5 (graphrag.py) ────┤
Task 6 (FSM fix) ────────┘
         │
Task 4 (log.py migration) ── depends on Tasks 1, 3 (same package)
         │
Task 7 (STT reconnect) ── independent
         │
Task 8 (small cleanups) ──┐
Task 9 (DRY cleanups) ────┘── last, after major refactoring
```

## Summary

| Task | Score | Lines Saved | Risk | Key Files |
|------|-------|-------------|------|-----------|
| 1. Simplify tracing.py | 4.4 | ~130 | LOW | logging/tracing.py |
| 2. Clean embeddings.py | 4.2 | ~30 | LOW | embeddings.py |
| 3. Dedup formatters.py | 3.6 | ~10 | LOW | logging/formatters.py |
| 4. Remove log.py facade | 3.4 | ~10 | LOW | log.py, 12 consumers |
| 5. GraphRAG conn helper | 3.2 | ~18 | LOW | graphrag.py |
| 6. Fix FSM encapsulation | 2.9 | ~4 | LOW | state.py, pipeline.py |
| 7. STT reconnect dedup | 2.6 | ~13 | MEDIUM | pipeline.py |
| 8. Small dead code | — | ~15 | LOW | llm.py, structured_logger.py, config.py |
| 9. DRY cleanups | — | ~30 | LOW | memory.py, hass.py, playback.py, pipeline.py |

**Total estimated lines removable:** ~260 (~7.1% of 3,646)
**Behavioral changes:** None — all simplifications preserve existing functionality
**Test updates required:** Yes — removed features need test cleanup (tracing tests, embedding tests, state tests)

---

## Review History

- **v1 (2026-02-19):** Initial 12-task plan, ~340 estimated lines
- **v2 (2026-02-19):** Revised after review — PASS WITH CHANGES applied:
  - DROPPED Task 6 (Share Anthropic clients) — adds coupling for negligible gain
  - DROPPED Task 8 (Voyage consolidation) — "reranker leak" was false (already closed via MemoryExtractor)
  - DROPPED Task 10 (_process_response refactoring) — too risky for simplification audit
  - CORRECTED Task 3 estimate from ~25 to ~10 lines
  - SHRUNK Task 4 to steps 1-2 only (log.py removal); dropped package restructuring
  - FIXED Task 6→7 (FSM) to list specific test files needing updates
  - FIXED Task 8→11 step 5: keep execute_tool, don't inline
  - ADDED note on Task 1 step 5 about async_gen asymmetry
