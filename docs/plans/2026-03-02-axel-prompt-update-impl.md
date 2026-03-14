# Axel Prompt Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use execute-flow to implement this plan task-by-task.

**Goal:** Update axel.xml with code-switching, dual-channel support, and wire channel info through context.py

**Architecture:** Single axel.xml with channel-scoped rules. ContextManager accepts a channel parameter, stores it, and injects it into dynamic block 3. Engine passes the channel through from its caller.

**Tech Stack:** Python, FastAPI, pytest

---

### Task 1: Add channel parameter to ContextManager

**Files:**
- Modify: `src/prot/context.py:19-52`
- Test: `tests/test_context.py`

**Step 1: Write the failing test**

Add to `tests/test_context.py`:

```python
def test_build_system_blocks_default_channel_is_voice(self):
    cm = ContextManager(persona_text="test persona", rag_context="")
    blocks = cm.build_system_blocks()
    assert "channel: voice" in blocks[2]["text"]

def test_build_system_blocks_chat_channel(self):
    cm = ContextManager(persona_text="test persona", rag_context="", channel="chat")
    blocks = cm.build_system_blocks()
    assert "channel: chat" in blocks[2]["text"]

def test_build_system_blocks_voice_channel_explicit(self):
    cm = ContextManager(persona_text="test persona", rag_context="", channel="voice")
    blocks = cm.build_system_blocks()
    assert "channel: voice" in blocks[2]["text"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context.py -v -k "channel"`
Expected: FAIL — `__init__() got an unexpected keyword argument 'channel'`

**Step 3: Implement channel parameter**

In `src/prot/context.py`, modify `__init__` and `build_system_blocks`:

```python
def __init__(self, persona_text: str, rag_context: str = "", channel: str = "voice") -> None:
    self._persona = persona_text
    self._rag_context = rag_context
    self._channel = channel
    self._messages: list[dict] = []
```

Update `build_system_blocks` block3:

```python
block3_dynamic: dict = {
    "type": "text",
    "text": (
        f"datetime: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"timezone: America/Vancouver\n"
        f"channel: {self._channel}"
    ),
}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_context.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/prot/context.py tests/test_context.py
git commit -m "feat: add channel parameter to ContextManager"
```

---

### Task 2: Wire channel through engine and callers

**Files:**
- Modify: `src/prot/pipeline.py:44`
- Modify: `src/prot/app.py:109-119`

**Step 1: Voice pipeline — explicit channel="voice"**

In `src/prot/pipeline.py`, line 44, the ContextManager is already created with
default channel="voice", so no change needed. But for clarity:

```python
self._ctx = ContextManager(persona_text=load_persona(), channel="voice")
```

**Step 2: Chat endpoint — create separate ContextManager with channel="chat"**

The chat WebSocket currently shares `pipeline._engine` which uses the voice
ContextManager. We need to give it a chat-specific context. Modify `app.py`
chat endpoint to create a per-connection engine with channel="chat":

```python
@app.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    """Text chat via shared ConversationEngine."""
    await websocket.accept()
    if not pipeline:
        await websocket.send_json({"type": "error", "message": "Server not ready"})
        await websocket.close()
        return

    # Chat gets its own context with channel="chat"
    ctx = ContextManager(persona_text=load_persona(), channel="chat")
    engine = ConversationEngine(
        ctx=ctx,
        llm=pipeline._llm,
        hass_agent=pipeline._hass_agent,
        memory=pipeline._memory,
        graphrag=pipeline._graphrag,
    )
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") != "message" or not data.get("content"):
                continue

            engine.add_user_message(data["content"])

            try:
                async for item in engine.respond():
                    if isinstance(item, ToolIterationMarker):
                        continue
                    await websocket.send_json({"type": "chunk", "content": item})

                result = engine.last_result
                await websocket.send_json(
                    {"type": "done", "full_text": result.full_text if result else ""}
                )
            except BusyError:
                await websocket.send_json(
                    {"type": "error", "message": "Engine is busy (voice active)"}
                )
            except Exception as exc:
                logger.exception("Chat respond error")
                await websocket.send_json(
                    {"type": "error", "message": str(exc)}
                )
    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception:
        logger.exception("Chat WebSocket error")
```

Add imports at top of `app.py`:

```python
from prot.context import ContextManager
from prot.engine import BusyError, ConversationEngine, ToolIterationMarker
from prot.persona import load_persona
```

Note: `BusyError` and `ToolIterationMarker` are already imported. Add
`ConversationEngine`, `ContextManager`, and `load_persona`.

**Step 3: Run existing tests**

Run: `uv run pytest tests/test_app.py tests/test_context.py -v`
Expected: ALL PASS (chat test may need updating since engine is now per-connection)

**Step 4: Update chat WebSocket test**

The test mocks `pipeline._engine` directly. With per-connection engines, we
need to mock `load_persona`, `ContextManager`, and `ConversationEngine`.
Update `tests/test_app.py` `TestChatWebSocket`:

```python
class TestChatWebSocket:
    def test_chat_sends_and_receives(self):
        """Sync test using Starlette TestClient for WebSocket."""
        mock_engine = MagicMock(spec=ConversationEngine)
        mock_engine.busy = False
        mock_engine.add_user_message = MagicMock()
        mock_engine.last_result = ResponseResult(
            full_text="Hello world", interrupted=False
        )

        async def fake_respond():
            yield "Hello "
            yield "world"

        mock_engine.respond = fake_respond

        mock_pipeline = MagicMock()
        mock_pipeline.startup = AsyncMock()
        mock_pipeline.shutdown = AsyncMock()
        mock_pipeline._engine = mock_engine
        mock_pipeline._llm = MagicMock()
        mock_pipeline._hass_agent = None
        mock_pipeline._memory = None
        mock_pipeline._graphrag = None
        mock_pipeline.current_state = "idle"

        mock_audio = MagicMock()
        mock_audio.start = MagicMock()
        mock_audio.stop = MagicMock()

        with patch("prot.app.Pipeline", return_value=mock_pipeline), \
             patch("prot.app.AudioManager", return_value=mock_audio), \
             patch("prot.app.ConversationEngine", return_value=mock_engine), \
             patch("prot.app.load_persona", return_value="test persona"):
            import prot.app as app_module
            app_module.pipeline = mock_pipeline
            app_module.audio = mock_audio

            client = TestClient(app_module.app)
            with client.websocket_connect("/chat") as ws:
                ws.send_json({"type": "message", "content": "hello"})
                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] in ("done", "error"):
                        break

            chunk_msgs = [m for m in messages if m["type"] == "chunk"]
            done_msg = [m for m in messages if m["type"] == "done"][0]

            assert len(chunk_msgs) == 2
            assert chunk_msgs[0]["content"] == "Hello "
            assert chunk_msgs[1]["content"] == "world"
            assert done_msg["full_text"] == "Hello world"

            mock_engine.add_user_message.assert_called_once_with("hello")

            # Cleanup
            app_module.pipeline = None
            app_module.audio = None
```

**Step 5: Run all tests**

Run: `uv run pytest tests/test_app.py tests/test_context.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/prot/pipeline.py src/prot/app.py tests/test_app.py
git commit -m "feat: wire channel parameter through pipeline and chat endpoint"
```

---

### Task 3: Rewrite axel.xml

**Files:**
- Modify: `data/axel.xml`

**Step 1: Replace axel.xml with the new prompt**

Full rewrite per design doc sections 2-7:
- `<identity>` — SGCE origin, English
- `<voice>` — updated rules with code-switching
- `<language>` — code-switching triggers and examples
- `<channels>` — voice and chat formatting/rhythm rules
- `<relationship>` — health intervention behavior added
- `<constraints>` — code-switching in always
- `<examples>` — all 7 rewritten with code-switching
- Remove stale `</output>` tag

**Step 2: Verify XML is well-formed**

Run: `python3 -c "import xml.etree.ElementTree as ET; ET.parse('data/axel.xml'); print('OK')"`
Expected: OK

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add data/axel.xml
git commit -m "feat: rewrite axel.xml with code-switching and dual-channel support"
```

---

### Task 4: Final verification

**Step 1: Run full test suite with coverage**

Run: `uv run pytest --cov=prot --cov-report=term-missing`
Expected: ALL PASS, no regressions

**Step 2: Verify prompt loads correctly**

Run: `python3 -c "from prot.persona import load_persona; p = load_persona(); print(len(p), 'chars'); assert '<channels>' in p; assert '<language' in p; print('OK')"`
Expected: OK

**Step 3: Verify dynamic block contains channel**

Run: `python3 -c "from prot.context import ContextManager; cm = ContextManager('test', channel='chat'); b = cm.build_system_blocks(); print(b[2]['text']); assert 'channel: chat' in b[2]['text']; print('OK')"`
Expected: prints block3 text with `channel: chat`, then OK
