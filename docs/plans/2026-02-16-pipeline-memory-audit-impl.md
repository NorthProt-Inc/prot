# Pipeline Data/Memory Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Systematically audit every data/memory pipeline in prot for leaks, unbounded growth, cleanup gaps, concurrency bugs, and data integrity issues — then fix all discovered problems with TDD.

**Architecture:** 3-phase approach — (1) Static code audit producing a findings report, (2) Test execution + new edge-case tests, (3) Runtime profiling instrumentation. Each finding gets a severity rating and targeted fix.

**Tech Stack:** Python 3.13, asyncio, pytest, pytest-cov, tracemalloc, asyncpg, websockets, PyAudio

---

## Phase 1: Static Code Audit (7-Axis Analysis)

### Task 1: Audit asyncio Task Lifecycle

**Files:**
- Read: `src/prot/pipeline.py:221-229` (create_task producer/consumer)
- Read: `src/prot/pipeline.py:272` (ensure_future timeout)
- Read: `src/prot/pipeline.py:294` (ensure_future memory extraction)
- Read: `src/prot/stt.py:57` (create_task recv_loop)
- Read: `src/prot/stt.py:86-100` (disconnect task cleanup)

**Step 1: Catalog all Task creation sites**

Inventory of every `create_task()` / `ensure_future()` call:

| Location | Purpose | Ref Stored? | Cancel Path? | Await/Callback? |
|----------|---------|-------------|--------------|-----------------|
| `pipeline.py:221` | `_produce()` task | `prod` local | `pending` cancel L226 | `asyncio.wait` L223 |
| `pipeline.py:222` | `_consume()` task | `cons` local | `pending` cancel L226 | `asyncio.wait` L223 |
| `pipeline.py:272` | `_timeout()` active timeout | `_active_timeout_task` | `shutdown()` L299 | No await |
| `pipeline.py:294` | `_extract()` memory bg | **NO** | **NO** | **NO** |
| `stt.py:57` | `_recv_loop()` | `_recv_task` | `disconnect()` L89 | await L91 |

**Step 2: Identify findings**

**FINDING-1 [High]: Fire-and-forget task in `_extract_memories_bg()` (pipeline.py:294)**
- `asyncio.ensure_future(_extract())` stores no reference
- If task is running during `shutdown()`, it continues as orphan
- If it raises, Python logs "Task exception was never retrieved"
- **Fix:** Store task refs in `_background_tasks: set[asyncio.Task]`, use `add_done_callback` for auto-removal, cancel all in `shutdown()`

**FINDING-2 [Low]: `_active_timeout_task` not awaited on cancel (pipeline.py:298-300)**
- `cancel()` is called but never `await`ed — the task may not have completed by the time `shutdown()` finishes
- **Fix:** `await` the task after cancel with `suppress(CancelledError)`

**Step 3: Commit audit notes (no code changes yet)**

This is analysis only. Findings are recorded in the report created in Task 8.

---

### Task 2: Audit Queue/Buffer Bounding

**Files:**
- Read: `src/prot/pipeline.py:169-201` (producer-consumer pipeline)
- Read: `src/prot/processing.py:14-30` (chunk_sentences)

**Step 1: Trace buffer growth path**

In `_process_response()`:
```python
buffer = ""          # L170 — starts empty
buffer += chunk      # L182 — grows with each LLM delta
sentences, buffer = chunk_sentences(buffer)  # L183 — resets on sentence boundary
```

**FINDING-3 [Medium]: Unbounded buffer accumulation (pipeline.py:170-183)**
- If LLM streams 1500 tokens without any `[.!?~]` character, `buffer` grows to ~6KB
- Korean text often uses `~` but edge cases exist (long lists, code blocks, etc.)
- The buffer is a local variable in `_process_response()`, so it's GC'd on function exit
- **Practical risk:** Low — Claude's max_tokens=1500 caps total output, and Korean almost always has terminators
- **Fix:** Add MAX_BUFFER guard that force-flushes if buffer exceeds threshold

**FINDING-4 [OK]: Queue backpressure is correctly implemented**
- `asyncio.Queue(maxsize=32)` at L171 — `put()` blocks when full
- Sentinel `None` sent in `finally` block (L200-201) — guaranteed even on exception
- Consumer breaks on `None` (L207-208) — clean exit
- **Verdict:** Correct

**FINDING-5 [Medium]: Queue not drained on barge-in**
- When `INTERRUPTED` is detected in consumer (L209-210), it breaks out of the loop
- But items already in the queue are abandoned (not consumed)
- These are `bytes` objects that will be GC'd when the queue goes out of scope (local var)
- **Practical risk:** Very low — queue is function-scoped, GC'd on `_process_response()` exit
- **Verdict:** Acceptable, no fix needed

---

### Task 3: Audit WebSocket/Stream Cleanup

**Files:**
- Read: `src/prot/stt.py:36-66` (connect/disconnect)
- Read: `src/prot/stt.py:67-84` (send_audio error handling)
- Read: `src/prot/tts.py:20-37` (stream_audio/flush)
- Read: `src/prot/llm.py:21-55` (stream_response/cancel)

**Step 1: Check STT connect/disconnect symmetry**

```
connect():
  1. disconnect()         — always cleans up first ✓
  2. websockets.connect() — opens new WS
  3. ws.recv()            — wait for session_started
  4. create_task(_recv_loop) — start recv

disconnect():
  1. _recv_task.cancel()  — cancel recv loop ✓
  2. await _recv_task      — wait for cancellation ✓
  3. _recv_task = None     — clear ref ✓
  4. ws.close()           — close WebSocket ✓
  5. _ws = None            — clear ref ✓
```

**FINDING-6 [Medium]: STT connect partial failure leaves _recv_task set (stt.py:36-65)**
- If `ws.recv()` (L51) raises after `websockets.connect()` succeeds, the except block at L58 closes `_ws` and sets it to None
- But `_recv_task` is set at L57 — since exception happens before L57, this is actually fine
- However, if `create_task(_recv_loop)` at L57 succeeds but the next line would fail... there's no next line, so this is fine
- **Verdict:** STT cleanup is correct

**FINDING-7 [OK]: TTS cancellation is correct**
- `flush()` sets `_cancelled = True`
- `stream_audio()` checks `_cancelled` each iteration and breaks
- Generator cleanup happens via Python's async generator finalization
- **Verdict:** Correct

**FINDING-8 [OK]: LLM stream cleanup is correct**
- `async with self._client.messages.stream(...)` — context manager handles cleanup
- `_active_stream = None` set after context manager exits (L51)
- `cancel()` sets `_cancelled = True`, loop breaks on next iteration
- **Verdict:** Correct

**Step 2: Check send_audio error recovery**

```python
send_audio():
  except:
    disconnect()     — but this is await inside sync-looking context
    fallback: _ws = None
```

**FINDING-9 [Low]: send_audio disconnect fallback (stt.py:79-84)**
- If `disconnect()` itself raises, `_ws = None` is set but `_recv_task` may be orphaned
- **Fix:** Set `_recv_task = None` in the same fallback block

---

### Task 4: Audit Thread↔Event Loop Safety

**Files:**
- Read: `src/prot/pipeline.py:92-99` (on_audio_chunk)
- Read: `src/prot/audio.py:44-54` (_audio_callback)
- Read: `src/prot/pipeline.py:49` (_loop storage)
- Read: `src/prot/pipeline.py:62` (_loop initialization)

**Step 1: Analyze thread-safety of shared state**

PyAudio callback thread calls `on_audio_chunk()` which calls `run_coroutine_threadsafe()`. This schedules `_async_audio_chunk()` on the event loop.

**Shared state accessed from PyAudio thread:**
- `self._loop` — read-only after startup(), set once at L62. **Thread-safe** (Python GIL + immutable after init)
- No other shared mutable state accessed in `on_audio_chunk()`

**Shared state accessed from event loop (via _async_audio_chunk):**
- `self._vad` — only accessed from event loop coroutines
- `self._sm` — only accessed from event loop coroutines
- `self._barge_in_count` — only accessed from event loop coroutines
- `self._speaking_since` — only accessed from event loop coroutines

**FINDING-10 [OK]: Thread safety is correct**
- `on_audio_chunk()` only reads `_loop` and calls `run_coroutine_threadsafe()`
- All mutable state is accessed only from the single event loop thread
- `RuntimeError` catch at L98 handles event loop shutdown during audio callback
- **Verdict:** Correct

**FINDING-11 [Low]: _loop could be None between __init__ and startup()**
- Guard at L94 handles this: `if self._loop is None: return`
- **Verdict:** Correct, guard exists

---

### Task 5: Audit DB Pool/Transaction Management

**Files:**
- Read: `src/prot/db.py:19-45` (pool lifecycle)
- Read: `src/prot/graphrag.py:16-18` (acquire)
- Read: `src/prot/graphrag.py:20-44` (upsert_entity with conn param)
- Read: `src/prot/memory.py:77-100` (save_extraction transaction)
- Read: `src/prot/embeddings.py:24-48` (semaphore)

**Step 1: Check acquire/release symmetry**

GraphRAG methods use two patterns:
1. **External conn provided** (called within a transaction): `await conn.fetchrow(...)` — caller manages lifecycle ✓
2. **Self-acquired**: `async with self._pool.acquire() as c:` — context manager auto-releases ✓

**FINDING-12 [OK]: All DB connections properly released**
- Every `self._pool.acquire()` uses `async with` context manager
- `memory.py:77-78` uses nested `async with` for both acquire and transaction
- Transaction rollback on exception is handled by asyncpg's transaction context manager
- **Verdict:** Correct

**FINDING-13 [OK]: Semaphore correctly limits concurrent embeddings**
- `embeddings.py:24` — `asyncio.Semaphore(max_concurrent=5)`
- Used as `async with self._semaphore:` around every API call
- **Verdict:** Correct

**FINDING-14 [Medium]: DB pool not closed if memory extraction is in-flight during shutdown**
- `pipeline.py:312-316` closes pool in `shutdown()`
- But `_extract_memories_bg()` fire-and-forget task may still be running
- If the task tries to use the pool after close → `asyncpg.InterfaceError`
- This is a consequence of FINDING-1 (fire-and-forget tasks)
- **Fix:** Same as FINDING-1 — track and cancel background tasks before closing pool

**FINDING-15 [Low]: Global pool singleton in db.py**
- `db.py:14` — `_pool: asyncpg.Pool | None = None` (module-level global)
- `init_pool()` raises if already initialized
- But pipeline.py doesn't use `db.init_pool()` directly — it stores the pool in `self._pool`
- The global `_pool` and pipeline's `self._pool` are the same object
- **Verdict:** Acceptable pattern, but cleanup only happens via pipeline.shutdown()

---

### Task 6: Audit Data Integrity

**Files:**
- Read: `src/prot/processing.py:1-30` (sentence chunking)
- Read: `src/prot/pipeline.py:143-161` (transcript accumulation)
- Read: `src/prot/pipeline.py:248-258` (barge-in handling)
- Read: `src/prot/state.py:1-77` (state transitions)

**Step 1: Verify sentence chunking correctness**

Regex: `_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?~])\s+')`

**FINDING-16 [Medium]: Sentence split regex doesn't handle Korean sentence endings**
- Korean commonly uses `다.`, `요.`, `야!` — these end with `[.!?]` and ARE handled
- But `~` is included as a sentence terminator — this means `"안녕~"` would be treated as a complete sentence ✓
- **Edge case:** Ellipsis `...` — splits after first `.`, producing fragments like `"안녕."` and `".."`
- **Edge case:** URLs like `example.com test` — the `.` splits mid-URL
- **Practical risk:** Low for voice conversations — URLs/ellipsis rare in spoken Korean
- **Verdict:** Acceptable for voice domain, but noted

**Step 2: Verify barge-in data integrity**

When barge-in happens:
1. `_llm.cancel()` — sets `_cancelled = True`, producer loop breaks
2. `_tts.flush()` — sets `_cancelled = True`, TTS generator breaks
3. `_player.kill()` — kills paplay process
4. `_current_transcript = ""` — clears transcript for next turn

**FINDING-17 [Medium]: Partial response not saved to context on barge-in**
- In `_process_response()`, `full_response` accumulates LLM text (L181)
- If interrupted, `try_on_tts_complete()` returns False (L234), so `add_message()` is never called
- The partial response is lost — user's next turn has no record of what was said
- **Practical impact:** Context continuity is broken, but this is arguably correct — incomplete responses shouldn't be in context
- **Verdict:** Acceptable design choice, but could add partial response as metadata

**FINDING-18 [OK]: State machine transitions are well-guarded**
- `_transition()` validates against `VALID_TRANSITIONS` dict
- Invalid transitions raise `ValueError`
- `try_on_tts_complete()` gracefully handles interrupted state (returns False)
- **Verdict:** Correct

---

### Task 7: Audit Process/Resource Cleanup

**Files:**
- Read: `src/prot/playback.py:28-78` (paplay lifecycle)
- Read: `src/prot/audio.py:22-58` (PyAudio lifecycle)
- Read: `src/prot/pipeline.py:296-317` (shutdown)
- Read: `src/prot/app.py:21-37` (lifespan)

**Step 1: Check paplay process management**

```
start():  kill previous (if alive) + wait ✓ → create new
play_chunk(): write + drain; on BrokenPipe → kill + wait ✓
finish(): close stdin + wait ✓ → process = None
kill():  kill + wait ✓ (in finally: process = None)
```

**FINDING-19 [OK]: paplay zombie prevention is correct**
- Every `kill()` is followed by `await self._process.wait()`
- `ProcessLookupError` caught (process already dead)
- `finally` block ensures `_process = None`
- **Verdict:** Correct

**Step 2: Check PyAudio cleanup**

```
start():  pa.open() + start_stream()
stop():   stop_stream() + close() → stream = None (in finally)
__del__(): stop() + pa.terminate()
```

**FINDING-20 [Low]: PyAudio relies on __del__ for pa.terminate()**
- `__del__` is called by GC, timing is unpredictable
- If the AudioManager object leaks (e.g., exception in app.py lifespan), `terminate()` may not be called promptly
- `app.py:35` calls `audio.stop()` but not `audio._pa.terminate()`
- **Fix:** Add explicit `terminate()` method called in lifespan shutdown, or add `_pa.terminate()` to `stop()`

**Step 3: Check shutdown order**

`app.py` lifespan shutdown:
```python
audio.stop()          # 1. Stop mic input
await pipeline.shutdown()  # 2. Shutdown pipeline
```

`pipeline.shutdown()`:
```python
cancel _active_timeout_task   # 1. Cancel timeout timer
await stt.disconnect()         # 2. Close STT WebSocket
await player.kill()            # 3. Kill paplay
await pool.close()             # 4. Close DB pool
```

**FINDING-21 [Medium]: Background memory extraction tasks not cancelled in shutdown (pipeline.py:296-317)**
- No reference to fire-and-forget tasks → can't cancel them
- They may try to use `_pool` after it's closed
- Same root cause as FINDING-1
- **Fix:** Track background tasks, cancel before pool close

**FINDING-22 [Low]: No signal handling in app.py**
- FastAPI's `lifespan` handles SIGTERM/SIGINT gracefully via uvicorn
- But if running without uvicorn, signals aren't caught
- **Verdict:** Acceptable — app is always run via uvicorn/FastAPI

---

## Phase 1 Summary: Findings

| ID | Severity | Component | Description |
|----|----------|-----------|-------------|
| F1 | **High** | pipeline.py:294 | Fire-and-forget `_extract_memories_bg()` — no ref, no cancel, no await |
| F2 | Low | pipeline.py:298-300 | `_active_timeout_task` not awaited after cancel |
| F3 | Medium | pipeline.py:170-183 | Unbounded LLM buffer (theoretical, capped by max_tokens) |
| F9 | Low | stt.py:79-84 | `send_audio` disconnect fallback doesn't clear `_recv_task` |
| F14 | Medium | pipeline.py:312-316 | DB pool closed while bg tasks may still use it |
| F16 | Medium | processing.py:3 | Ellipsis/URL edge cases in sentence split regex |
| F17 | Medium | pipeline.py:231-239 | Partial response lost on barge-in (design choice) |
| F20 | Low | audio.py:56-58 | PyAudio terminate relies on `__del__` |
| F21 | Medium | pipeline.py:296-317 | Shutdown doesn't cancel background tasks |

**Critical fixes needed:** F1 + F14 + F21 (all same root cause — background task management)

---

## Phase 2: Test Execution & New Tests

### Task 8: Run Existing Test Suite

**Step 1: Run all tests**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/ -v --tb=short 2>&1 | head -100`
Expected: All tests pass (or identify failures)

**Step 2: Run with coverage**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/ --cov=prot --cov-report=term-missing --tb=short 2>&1 | head -100`
Expected: Coverage report showing uncovered lines

**Step 3: Document failures and coverage gaps**

Record any failing tests and uncovered code paths relevant to data/memory.

---

### Task 9: Write Tests for F1 — Background Task Tracking

**Files:**
- Test: `tests/test_pipeline.py`
- Modify: `src/prot/pipeline.py`

**Step 1: Write failing test for background task tracking**

```python
class TestExtractMemoriesBg:
    """_extract_memories_bg() — background task lifecycle."""

    async def test_background_task_tracked(self):
        """Fire-and-forget tasks should be tracked for shutdown cleanup."""
        p = _make_pipeline()
        # Set up memory mock
        mock_memory = AsyncMock()
        mock_memory.extract_from_conversation = AsyncMock(
            return_value={"entities": [], "relationships": []}
        )
        mock_memory.save_extraction = AsyncMock()
        p._memory = mock_memory

        p._extract_memories_bg()
        # Task should be tracked
        assert len(p._background_tasks) == 1
        # Let it complete
        await asyncio.sleep(0.05)
        # Completed tasks should auto-remove
        assert len(p._background_tasks) == 0

    async def test_shutdown_cancels_background_tasks(self):
        """shutdown() should cancel in-flight background tasks."""
        p = _make_pipeline()

        # Create a slow background task
        async def slow_extract():
            await asyncio.sleep(10)

        mock_memory = AsyncMock()
        mock_memory.extract_from_conversation = slow_extract
        p._memory = mock_memory

        p._extract_memories_bg()
        assert len(p._background_tasks) == 1

        await p.shutdown()
        # All background tasks should be cancelled
        assert len(p._background_tasks) == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py::TestExtractMemoriesBg -v`
Expected: FAIL with `AttributeError: 'Pipeline' has no attribute '_background_tasks'`

**Step 3: Implement background task tracking in pipeline.py**

Add to `__init__()` (after L53):
```python
self._background_tasks: set[asyncio.Task] = set()
```

Replace `_extract_memories_bg()` (L274-294):
```python
def _extract_memories_bg(self) -> None:
    """Background task to extract and save memories — tracked for shutdown cleanup."""
    if not self._memory:
        return

    async def _extract() -> None:
        try:
            messages = self._ctx.get_messages()
            extraction = await self._memory.extract_from_conversation(messages)
            await self._memory.save_extraction(extraction)

            # Refresh RAG context
            if self._current_transcript:
                rag = await self._memory.pre_load_context(
                    self._current_transcript
                )
                self._ctx.update_rag_context(rag)
        except Exception:
            logger.debug("Memory extraction failed", exc_info=True)

    task = asyncio.create_task(_extract())
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)
```

Update `shutdown()` — add before pool close (before L312):
```python
# Cancel background tasks
for task in self._background_tasks:
    task.cancel()
if self._background_tasks:
    await asyncio.gather(*self._background_tasks, return_exceptions=True)
self._background_tasks.clear()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py::TestExtractMemoriesBg -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/prot/pipeline.py tests/test_pipeline.py
git commit -m "fix(pipeline): track background tasks for proper shutdown cleanup"
```

---

### Task 10: Write Tests for F2 — Await Timeout Task on Cancel

**Files:**
- Test: `tests/test_pipeline.py`
- Modify: `src/prot/pipeline.py`

**Step 1: Write failing test**

```python
class TestShutdownAwaitTimeout:
    async def test_shutdown_awaits_timeout_task(self):
        """shutdown() should await the timeout task after cancelling it."""
        p = _make_pipeline()
        # Create a real timeout task
        async def _slow_timeout():
            await asyncio.sleep(100)

        p._active_timeout_task = asyncio.create_task(_slow_timeout())

        await p.shutdown()
        assert p._active_timeout_task is None
        # The task should be done (cancelled)
        # No warnings about "Task was destroyed but it is pending"
```

**Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pipeline.py::TestShutdownAwaitTimeout -v`
Expected: May pass but with ResourceWarning about unawaited task

**Step 3: Implement proper await in shutdown**

Replace in `shutdown()` (L297-300):
```python
if self._active_timeout_task is not None:
    self._active_timeout_task.cancel()
    try:
        await self._active_timeout_task
    except (asyncio.CancelledError, Exception):
        pass
    self._active_timeout_task = None
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py::TestShutdownAwaitTimeout -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All pass

**Step 6: Commit**

```bash
git add src/prot/pipeline.py tests/test_pipeline.py
git commit -m "fix(pipeline): await timeout task on shutdown to prevent resource warnings"
```

---

### Task 11: Write Tests for F3 — Buffer Overflow Guard

**Files:**
- Test: `tests/test_processing.py`
- Modify: `src/prot/processing.py`

**Step 1: Write failing test for max buffer**

```python
class TestChunkSentencesBufferGuard:
    def test_force_flush_on_oversized_input(self):
        """chunk_sentences should force-flush text exceeding MAX_BUFFER_CHARS."""
        from prot.processing import MAX_BUFFER_CHARS
        long_text = "가" * (MAX_BUFFER_CHARS + 100)  # no sentence terminators
        sentences, remainder = chunk_sentences(long_text)
        # Should have flushed something rather than keeping it all in remainder
        assert len(remainder) <= MAX_BUFFER_CHARS
        assert len(sentences) >= 1
```

**Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_processing.py::TestChunkSentencesBufferGuard -v`
Expected: FAIL — `ImportError: cannot import name 'MAX_BUFFER_CHARS'`

**Step 3: Implement buffer guard**

Add to `processing.py` (after L4):
```python
MAX_BUFFER_CHARS = 2000
```

Update `chunk_sentences()`:
```python
def chunk_sentences(text: str) -> tuple[list[str], str]:
    """Split text into complete sentences and a trailing remainder.

    Returns (complete_sentences, remainder) where remainder is the
    trailing text that does not end with a sentence terminator.
    If the remainder exceeds MAX_BUFFER_CHARS, force-flush it as a sentence.
    """
    stripped = text.strip()
    if not stripped:
        return [], ""
    parts = _RE_SENTENCE_SPLIT.split(stripped)
    if not parts:
        return [], stripped
    if _RE_SENTENCE_END.search(parts[-1]):
        return [p.strip() for p in parts if p.strip()], ""
    remainder = parts.pop().strip()
    complete = [p.strip() for p in parts if p.strip()]
    if len(remainder) > MAX_BUFFER_CHARS:
        complete.append(remainder)
        remainder = ""
    return complete, remainder
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_processing.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/prot/processing.py tests/test_processing.py
git commit -m "fix(processing): add MAX_BUFFER_CHARS guard to prevent unbounded buffer growth"
```

---

### Task 12: Write Tests for F9 — STT send_audio Fallback Cleanup

**Files:**
- Test: `tests/test_stt.py`
- Modify: `src/prot/stt.py`

**Step 1: Write failing test**

```python
class TestSendAudioFallbackCleanup:
    async def test_disconnect_failure_clears_recv_task(self):
        """If disconnect() fails in send_audio fallback, _recv_task should still be cleared."""
        client = STTClient(api_key="test")
        client._ws = AsyncMock()
        client._ws.send = AsyncMock(side_effect=Exception("send failed"))
        mock_task = MagicMock()
        client._recv_task = mock_task

        # disconnect() itself will raise
        client.disconnect = AsyncMock(side_effect=Exception("disconnect failed"))

        await client.send_audio(b"\x00" * 100)
        assert client._ws is None
        assert client._recv_task is None  # This will fail before fix
```

**Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_stt.py::TestSendAudioFallbackCleanup -v`
Expected: FAIL — `_recv_task` is not None

**Step 3: Fix send_audio fallback**

Replace `send_audio` except block (stt.py:79-84):
```python
        except Exception:
            logger.warning("Send failed, disconnecting")
            try:
                await self.disconnect()
            except Exception:
                self._ws = None
                self._recv_task = None
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stt.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/prot/stt.py tests/test_stt.py
git commit -m "fix(stt): clear _recv_task in send_audio disconnect fallback"
```

---

### Task 13: Write Test for F20 — PyAudio Explicit Terminate

**Files:**
- Test: `tests/test_audio.py`
- Modify: `src/prot/audio.py`

**Step 1: Write failing test**

```python
class TestAudioManagerTerminate:
    def test_stop_terminates_pyaudio(self):
        """stop() should call pa.terminate() for deterministic cleanup."""
        with patch("prot.audio.pyaudio.PyAudio") as MockPA:
            mock_pa = MockPA.return_value
            am = AudioManager(on_audio=lambda x: None)
            am.stop()
            mock_pa.terminate.assert_called_once()
```

**Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_audio.py::TestAudioManagerTerminate -v`
Expected: FAIL — `terminate()` not called in `stop()`

**Step 3: Add terminate to stop()**

Update `audio.py` `stop()` method:
```python
def stop(self) -> None:
    """Stop and close mic stream."""
    if self._stream:
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._stream = None
    self._pa.terminate()
```

Remove `__del__` method (no longer needed — stop() handles everything).

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/prot/audio.py tests/test_audio.py
git commit -m "fix(audio): add explicit pa.terminate() in stop() instead of relying on __del__"
```

---

## Phase 3: Runtime Profiling Instrumentation

### Task 14: Add Diagnostic Health Endpoint

**Files:**
- Modify: `src/prot/pipeline.py`
- Modify: `src/prot/app.py`

**Step 1: Add diagnostics method to Pipeline**

Add to `Pipeline` class:
```python
def diagnostics(self) -> dict:
    """Return runtime diagnostics for monitoring."""
    import asyncio
    diag = {
        "state": self._sm.state.value,
        "background_tasks": len(self._background_tasks),
        "active_timeout": self._active_timeout_task is not None,
        "asyncio_tasks": len(asyncio.all_tasks()),
    }
    if self._pool is not None:
        diag["db_pool_size"] = self._pool.get_size()
        diag["db_pool_free"] = self._pool.get_idle_size()
    return diag
```

**Step 2: Add /diagnostics endpoint to app.py**

```python
@app.get("/diagnostics")
async def diagnostics():
    if not pipeline:
        return {"status": "not_started"}
    return pipeline.diagnostics()
```

**Step 3: Commit**

```bash
git add src/prot/pipeline.py src/prot/app.py
git commit -m "feat(pipeline): add /diagnostics endpoint for runtime monitoring"
```

---

### Task 15: Add tracemalloc Profiling Support

**Files:**
- Modify: `src/prot/app.py`

**Step 1: Add tracemalloc startup option**

Add to `app.py` lifespan (before Pipeline init):
```python
import tracemalloc
import os
if os.environ.get("PROT_TRACEMALLOC"):
    tracemalloc.start()
    logger.info("tracemalloc enabled")
```

**Step 2: Add /memory endpoint**

```python
@app.get("/memory")
async def memory_stats():
    import tracemalloc
    if not tracemalloc.is_tracing():
        return {"error": "tracemalloc not enabled. Set PROT_TRACEMALLOC=1"}
    snapshot = tracemalloc.take_snapshot()
    top = snapshot.statistics("lineno")[:20]
    return {
        "current_mb": tracemalloc.get_traced_memory()[0] / 1024 / 1024,
        "peak_mb": tracemalloc.get_traced_memory()[1] / 1024 / 1024,
        "top_allocations": [
            {"file": str(s.traceback), "size_kb": s.size / 1024}
            for s in top
        ],
    }
```

**Step 3: Commit**

```bash
git add src/prot/app.py
git commit -m "feat(app): add tracemalloc profiling and /memory endpoint"
```

---

### Task 16: Add Queue Size Logging to Pipeline

**Files:**
- Modify: `src/prot/pipeline.py`

**Step 1: Add queue size logging in producer**

In `_produce()`, after `await audio_q.put(audio)` (L191):
```python
if audio_q.qsize() >= 24:  # 75% full
    logger.warning("Queue pressure", qsize=audio_q.qsize(), maxsize=32)
```

**Step 2: Commit**

```bash
git add src/prot/pipeline.py
git commit -m "feat(pipeline): add queue pressure warning logging"
```

---

### Task 17: Final Audit Report & Verification

**Files:**
- Create: `docs/plans/2026-02-16-pipeline-memory-audit-report.md`

**Step 1: Run full test suite with coverage**

Run: `python -m pytest tests/ --cov=prot --cov-report=term-missing -v`
Expected: All pass, coverage ≥ 85%

**Step 2: Write final report summarizing all findings and fixes**

Document:
- All findings (F1-F22) with status (fixed/accepted/deferred)
- Test coverage before and after
- Runtime profiling endpoints added
- Remaining risks and monitoring recommendations

**Step 3: Commit report**

```bash
git add docs/plans/2026-02-16-pipeline-memory-audit-report.md
git commit -m "docs: add pipeline memory audit final report"
```

---

## Task Dependency Graph

```
Task 1-7 (audit)     → Task 8 (run tests)
Task 8               → Task 9-13 (fixes, parallel)
Task 9-13            → Task 14-16 (profiling, parallel)
Task 14-16           → Task 17 (final report)
```

## Summary of Code Changes

| File | Change | Reason |
|------|--------|--------|
| `pipeline.py` | Add `_background_tasks` set, update `_extract_memories_bg()`, update `shutdown()` | F1, F14, F21 |
| `pipeline.py` | Await `_active_timeout_task` in shutdown | F2 |
| `pipeline.py` | Add `diagnostics()` method | Monitoring |
| `pipeline.py` | Add queue pressure logging | Monitoring |
| `processing.py` | Add `MAX_BUFFER_CHARS`, guard in `chunk_sentences()` | F3 |
| `stt.py` | Clear `_recv_task` in send_audio fallback | F9 |
| `audio.py` | Add `pa.terminate()` to `stop()`, remove `__del__` | F20 |
| `app.py` | Add `/diagnostics`, `/memory` endpoints, tracemalloc | Monitoring |
