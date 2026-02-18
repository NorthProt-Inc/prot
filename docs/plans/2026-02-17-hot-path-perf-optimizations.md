# Hot-Path Performance Optimizations

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize the three validated performance bottlenecks on the voice pipeline hot path.

**Architecture:** Parallelize sequential DB queries in memory pre-loading, pre-compute the STT audio message template to eliminate per-chunk JSON serialization, and remove dead code.

**Tech Stack:** Python asyncio, PostgreSQL (asyncpg), WebSocket (websockets), torch

---

### Task 1: Parallelize N+1 neighbor queries in `MemoryExtractor.pre_load_context`

**Files:**
- Modify: `src/prot/memory.py:130-142` (`pre_load_context` method)
- Test: `tests/test_memory.py`

**Context:**
Currently `pre_load_context` calls `get_entity_neighbors()` sequentially for each of the 5 entities returned by semantic search. Each call is an independent DB query (recursive CTE). These can run concurrently with `asyncio.gather`.

The tricky part: the token budget accumulator (`_add`) must still respect ordering and budget limits. So we gather all neighbor results first, then iterate in order.

**Step 1: Write the failing test**

Add to `tests/test_memory.py`:

```python
async def test_pre_load_context_fetches_neighbors_concurrently(self):
    """Verify neighbor queries run concurrently, not sequentially."""
    import asyncio

    call_order = []

    async def mock_get_neighbors(entity_id, max_depth=1):
        call_order.append(("start", entity_id))
        await asyncio.sleep(0.01)  # simulate DB latency
        call_order.append(("end", entity_id))
        return [{"name": f"Neighbor-of-{entity_id}", "description": "neighbor"}]

    mock_store = AsyncMock()
    mock_store.search_entities_semantic.return_value = [
        {"id": f"e{i}", "name": f"Entity{i}", "entity_type": "person",
         "description": f"Desc {i}"}
        for i in range(3)
    ]
    mock_store.get_entity_neighbors = mock_get_neighbors
    mock_store.search_communities.return_value = []
    mock_embedder = AsyncMock()
    mock_embedder.embed_query.return_value = [0.1] * 1024

    extractor = MemoryExtractor(
        anthropic_key="test", store=mock_store, embedder=mock_embedder
    )
    text = await extractor.pre_load_context("test query")

    # If concurrent: all "start" before any "end"
    # If sequential: start/end/start/end/start/end
    starts = [i for i, (action, _) in enumerate(call_order) if action == "start"]
    ends = [i for i, (action, _) in enumerate(call_order) if action == "end"]
    # All starts should come before all ends
    assert max(starts) < min(ends), (
        f"Neighbor queries appear sequential: {call_order}"
    )

    # Results should still be in output
    assert "Entity0" in text
    assert "Neighbor-of-e0" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory.py::TestMemoryExtractor::test_pre_load_context_fetches_neighbors_concurrently -v`
Expected: FAIL — `max(starts) < min(ends)` assertion fails because current code is sequential

**Step 3: Implement concurrent neighbor fetching**

In `src/prot/memory.py`, replace `pre_load_context` body. The key change:
1. Gather all neighbor queries concurrently with `asyncio.gather`
2. Then iterate entities + their pre-fetched neighbors in order, respecting the token budget

```python
async def pre_load_context(self, query: str) -> str:
    """Search GraphRAG (entities + neighbors + communities) and assemble Block 2 context."""
    query_embedding = await self._embedder.embed_query(query)

    parts: list[str] = []
    token_estimate = 0

    def _add(text: str) -> bool:
        nonlocal token_estimate
        token_estimate += len(text) // 4
        if token_estimate > settings.rag_context_target_tokens:
            return False
        parts.append(text)
        return True

    # 1. Entity semantic search + neighbor traversal (concurrent)
    entities = await self._store.search_entities_semantic(
        query_embedding=query_embedding, top_k=5,
    )
    neighbor_lists = await asyncio.gather(*(
        self._store.get_entity_neighbors(entity["id"], max_depth=1)
        for entity in entities
    ))
    for entity, neighbors in zip(entities, neighbor_lists):
        line = f"- {entity['name']} ({entity['entity_type']}): {entity['description']}"
        if not _add(line):
            break
        for n in neighbors[:3]:
            nline = f"  > {n['name']}: {n['description']}"
            if not _add(nline):
                break

    # 2. Community summaries
    communities = await self._store.search_communities(
        query_embedding=query_embedding,
        top_k=settings.rag_top_k,
    )
    for community in communities:
        if not _add(community["summary"]):
            break

    return "\n".join(parts) if parts else "(no memory context)"
```

Note: `import asyncio` must be at the top of the file.

**Step 4: Run all memory tests**

Run: `pytest tests/test_memory.py -v`
Expected: All PASS (including existing tests — behavior is identical, only execution order changes)

**Step 5: Commit**

```bash
git add src/prot/memory.py tests/test_memory.py
git commit -m "perf: parallelize neighbor queries in pre_load_context"
```

---

### Task 2: Pre-compute STT audio message template

**Files:**
- Modify: `src/prot/stt.py:97-115` (`STTClient.__init__` and `send_audio`)
- Test: `tests/test_stt.py`

**Context:**
`send_audio` is called ~31 times/second (every 32ms). Currently it builds a full JSON dict and calls `json.dumps` every time. The only variable part is the base64-encoded audio data — all other fields are static. Pre-computing the message prefix/suffix eliminates `json.dumps` overhead on the hot path.

**Step 1: Write the failing test**

Add to `tests/test_stt.py`:

```python
async def test_send_audio_uses_precomputed_template(self):
    """Verify send_audio does not call json.dumps on the hot path."""
    ws = _make_ws_mock()
    with patch("prot.stt.websockets.connect", AsyncMock(return_value=ws)):
        client = STTClient(api_key="test")
        await client.connect()

        with patch("prot.stt.json.dumps") as mock_dumps:
            await client.send_audio(b"\x00" * 512)
            mock_dumps.assert_not_called()

        # Verify the sent message is still valid JSON with correct fields
        call_args = ws.send.call_args[0][0]
        msg = json.loads(call_args)
        assert msg["message_type"] == "input_audio_chunk"
        assert msg["sample_rate"] == 16000
        assert msg["commit"] is False
        raw = base64.b64decode(msg["audio_base_64"])
        assert raw == b"\x00" * 512

        await client.disconnect()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_stt.py::TestSTTClient::test_send_audio_uses_precomputed_template -v`
Expected: FAIL — `json.dumps` is still called in `send_audio`

**Step 3: Implement pre-computed template**

In `src/prot/stt.py`, add template fields to `__init__` and update `send_audio`:

In `__init__`, after existing fields:
```python
self._msg_prefix = '{"message_type":"input_audio_chunk","audio_base_64":"'
self._msg_suffix = f'","commit":false,"sample_rate":{settings.sample_rate}}}'
```

Replace `send_audio` body:
```python
async def send_audio(self, data: bytes) -> None:
    """Send PCM audio chunk to ElevenLabs."""
    if self._ws is None:
        return
    try:
        msg = self._msg_prefix + base64.b64encode(data).decode() + self._msg_suffix
        await self._ws.send(msg)
    except Exception:
        logger.warning("Send failed, disconnecting")
        try:
            await self.disconnect()
        except Exception:
            self._ws = None
            self._recv_task = None
```

**Step 4: Run all STT tests**

Run: `pytest tests/test_stt.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/prot/stt.py tests/test_stt.py
git commit -m "perf: pre-compute STT audio message template to avoid json.dumps on hot path"
```

---

### Task 3: Remove dead code `ensure_complete_sentence`

**Files:**
- Modify: `src/prot/processing.py:9-13`
- Test: `tests/test_processing.py` (verify no test references it)

**Context:**
`ensure_complete_sentence` is defined in `processing.py` but never imported or called anywhere in the codebase. Only `chunk_sentences` is used (by `pipeline.py`).

**Step 1: Verify no references exist**

Run: `grep -r "ensure_complete_sentence" src/ tests/`
Expected: Only the definition in `processing.py` and possibly a test in `test_processing.py`

**Step 2: Check test file for references**

Read `tests/test_processing.py` to see if there are tests for `ensure_complete_sentence`.
If tests exist, remove them. If no tests exist, proceed.

**Step 3: Remove the function**

Delete `ensure_complete_sentence` function (lines 9-13) from `src/prot/processing.py`.
Remove any tests that reference it from `tests/test_processing.py`.

**Step 4: Run all processing tests**

Run: `pytest tests/test_processing.py -v`
Expected: All remaining PASS

**Step 5: Run full test suite**

Run: `pytest -v`
Expected: All PASS — no import errors, no broken references

**Step 6: Commit**

```bash
git add src/prot/processing.py tests/test_processing.py
git commit -m "refactor: remove unused ensure_complete_sentence"
```
