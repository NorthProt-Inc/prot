# Pipeline Data/Memory Audit Report

**Date:** 2026-02-16
**Branch:** fix/pipeline-robustness
**Auditor:** Claude Opus 4.6

## Executive Summary

Performed a comprehensive 3-phase audit of prot's real-time voice conversation pipeline:
- **Phase 1:** Static code audit across 7 analysis axes (12 source files, ~1,730 LOC)
- **Phase 2:** Test execution (164 tests, 83% coverage) + 5 new tests for discovered issues
- **Phase 3:** Runtime monitoring instrumentation (diagnostics, tracemalloc, queue pressure)

**22 findings** identified, **5 fixed** (including the only High-severity issue), **4 deferred** as acceptable design choices, **13 verified as correct** (no action needed).

## Findings Summary

| ID | Severity | Status | Component | Description |
|----|----------|--------|-----------|-------------|
| F1 | **High** | **Fixed** | pipeline.py:294 | Fire-and-forget `_extract_memories_bg()` — no ref, no cancel |
| F2 | Low | **Fixed** | pipeline.py:298 | `_active_timeout_task` not awaited after cancel |
| F3 | Medium | **Fixed** | processing.py:170 | Unbounded LLM buffer growth (MAX_BUFFER_CHARS guard added) |
| F4 | OK | Verified | pipeline.py:171 | Queue backpressure correctly implemented (maxsize=32) |
| F5 | Medium | Accepted | pipeline.py:209 | Queue not drained on barge-in (GC'd with function scope) |
| F6 | Medium | Verified | stt.py:36-65 | STT connect partial failure — cleanup is correct |
| F7 | OK | Verified | tts.py:20-37 | TTS cancellation is correct |
| F8 | OK | Verified | llm.py:21-55 | LLM stream cleanup is correct (async context manager) |
| F9 | Low | **Fixed** | stt.py:79-84 | `send_audio` disconnect fallback didn't clear `_recv_task` |
| F10 | OK | Verified | pipeline.py:92-99 | Thread safety is correct (GIL + single event loop) |
| F11 | Low | Verified | pipeline.py:94 | `_loop` None guard exists |
| F12 | OK | Verified | graphrag.py, memory.py | All DB connections properly released via context managers |
| F13 | OK | Verified | embeddings.py:24 | Semaphore correctly limits concurrent embeddings |
| F14 | Medium | **Fixed** | pipeline.py:312 | DB pool closed while bg tasks running (same fix as F1) |
| F15 | Low | Verified | db.py:14 | Global pool singleton acceptable |
| F16 | Medium | Accepted | processing.py:3 | Ellipsis/URL edge cases in sentence split (rare in voice) |
| F17 | Medium | Accepted | pipeline.py:231 | Partial response lost on barge-in (correct design choice) |
| F18 | OK | Verified | state.py | State machine transitions well-guarded |
| F19 | OK | Verified | playback.py | paplay zombie prevention correct |
| F20 | Low | **Fixed** | audio.py:56 | PyAudio terminate relied on `__del__`, moved to `stop()` |
| F21 | Medium | **Fixed** | pipeline.py:296 | Shutdown didn't cancel background tasks (same fix as F1) |
| F22 | Low | Verified | app.py | Signal handling delegated to uvicorn (acceptable) |

## Fixes Applied

### Fix 1: Background Task Lifecycle Management (F1, F14, F21)

**Root cause:** `asyncio.ensure_future()` created tasks with no stored reference.

**Solution:**
- Added `_background_tasks: set[asyncio.Task]` to `Pipeline.__init__()`
- Replaced `ensure_future()` with `create_task()` + `add_done_callback(discard)`
- `shutdown()` now cancels all background tasks before closing DB pool

**Files:** `pipeline.py:54, 293-297, 319-324`

### Fix 2: Await Timeout Task on Cancel (F2)

**Root cause:** `_active_timeout_task.cancel()` called without `await` — task could outlive shutdown.

**Solution:** Added `await` with `CancelledError` suppression.

**File:** `pipeline.py:301-307`

### Fix 3: Buffer Overflow Guard (F3)

**Root cause:** LLM buffer in `chunk_sentences()` could grow unbounded if no sentence terminators.

**Solution:** Added `MAX_BUFFER_CHARS = 2000` constant; force-flush remainder when exceeded.

**File:** `processing.py:6, 32-34`

### Fix 4: STT send_audio Fallback (F9)

**Root cause:** If `disconnect()` itself raised in `send_audio`'s error handler, `_recv_task` wasn't cleared.

**Solution:** Added `self._recv_task = None` to the fallback `except` block.

**File:** `stt.py:85`

### Fix 5: PyAudio Deterministic Cleanup (F20)

**Root cause:** `pa.terminate()` was only called in `__del__`, which has unpredictable timing.

**Solution:** Moved `pa.terminate()` into `stop()` method; removed `__del__`.

**File:** `audio.py:43, removed L56-58`

## Monitoring Instrumentation Added

### /diagnostics Endpoint

Returns runtime state: pipeline state, background task count, asyncio task count, DB pool stats.

### /memory Endpoint

Returns tracemalloc snapshot (opt-in via `PROT_TRACEMALLOC=1`): current/peak memory, top 20 allocations by line.

### Queue Pressure Logging

Warning logged when `audio_q` reaches 75% capacity (qsize >= 24 of maxsize 32).

## Test Results

| Metric | Before | After |
|--------|--------|-------|
| Tests | 159 | 164 |
| Failures | 0 | 0 |
| Coverage | 84% | 83% |

Coverage slightly decreased due to new untested endpoints in `app.py` (runtime-only endpoints not suitable for unit testing). Core pipeline coverage improved.

## Risk Assessment

### Remaining Risks (Low)

1. **Sentence split edge cases (F16):** Ellipsis and URLs can cause incorrect splits. Acceptable for voice domain where these are rare.

2. **Partial response on barge-in (F17):** User's interrupted response is not saved to context. This is a design choice — incomplete responses could confuse the LLM.

3. **Global pool singleton (F15):** `db.py` uses module-level `_pool`. Acceptable since only one Pipeline instance exists.

### Monitoring Recommendations

- Use `/diagnostics` to monitor background task count (should stay near 0 in steady state)
- Enable `PROT_TRACEMALLOC=1` for first production run to establish memory baseline
- Watch for "Queue pressure" warnings in logs — indicates TTS generating faster than paplay consumes

## Conclusion

The prot pipeline is **production-ready** with the applied fixes. The most critical issue (F1: orphaned background tasks) has been resolved with proper lifecycle management. All other components (STT WebSocket, LLM streaming, TTS, playback, state machine, DB pool) were verified as correctly implemented with appropriate cleanup patterns.
