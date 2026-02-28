# HA Agent Delegation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use execute-flow to implement this plan task-by-task.

**Goal:** Replace prot's `HassRegistry` (2-tool, enum-based HA integration) with `HassAgent` (single `hass_request` tool delegating to HA's conversation API).

**Architecture:** `HassAgent` sends natural language commands to HA's `/api/conversation/process` endpoint. Claude interprets user intent, formulates a clear command string, and HA's Gemini Flash agent handles entity resolution and service calls. Results flow back through the existing tool loop.

**Tech Stack:** httpx (async HTTP), HA conversation API, pytest + pytest-asyncio

---

### Task 1: Write HassAgent tests

**Files:**
- Rewrite: `tests/test_hass.py`

**Step 1: Write the failing tests**

Replace the entire `tests/test_hass.py` with:

```python
"""Tests for HassAgent — HA conversation API delegation."""

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prot.hass import HassAgent


class TestHassAgentInit:
    def test_strips_trailing_slash(self):
        agent = HassAgent("http://hass:8123/", "token")
        assert agent._url == "http://hass:8123"

    def test_stores_token(self):
        agent = HassAgent("http://hass:8123", "my-token")
        assert agent._token == "my-token"


class TestHassAgentBuildTool:
    def test_returns_valid_tool_schema(self):
        agent = HassAgent("http://hass:8123", "token")
        tool = agent.build_tool()
        assert tool["name"] == "hass_request"
        assert "input_schema" in tool
        props = tool["input_schema"]["properties"]
        assert "command" in props
        assert tool["input_schema"]["required"] == ["command"]

    def test_schema_has_description(self):
        agent = HassAgent("http://hass:8123", "token")
        tool = agent.build_tool()
        assert "description" in tool
        assert len(tool["description"]) > 0


class TestHassAgentRequest:
    async def test_success_returns_speech(self):
        agent = HassAgent("http://hass:8123", "token")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": {
                "speech": {
                    "plain": {"speech": "거실 조명을 켰습니다"}
                }
            }
        }
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)

        result = await agent.request("거실 조명 켜줘")

        assert result == "거실 조명을 켰습니다"
        call_args = agent._client.post.call_args
        assert "/api/conversation/process" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["text"] == "거실 조명 켜줘"
        assert body["language"] == "ko"

    async def test_posts_with_auth_header(self):
        agent = HassAgent("http://hass:8123", "my-token")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": {"speech": {"plain": {"speech": "ok"}}}
        }
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)

        await agent.request("test")

        headers = agent._client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-token"

    async def test_connection_error_returns_message(self):
        agent = HassAgent("http://hass:8123", "token")
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await agent.request("test")
        assert "연결할 수 없습니다" in result

    async def test_auth_error_returns_message(self):
        agent = HassAgent("http://hass:8123", "token")
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)

        result = await agent.request("test")
        assert "인증 실패" in result

    async def test_timeout_returns_message(self):
        agent = HassAgent("http://hass:8123", "token")
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("timed out")
        )

        result = await agent.request("test")
        assert "시간 초과" in result

    async def test_generic_http_error_returns_message(self):
        agent = HassAgent("http://hass:8123", "token")
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)

        result = await agent.request("test")
        assert "오류" in result or "실패" in result


class TestHassAgentClose:
    async def test_close_calls_aclose(self):
        agent = HassAgent("http://hass:8123", "token")
        agent._client = AsyncMock()
        await agent.close()
        agent._client.aclose.assert_awaited_once()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hass.py -v`
Expected: FAIL — `HassAgent` does not exist yet.

**Step 3: Commit**

```bash
git add tests/test_hass.py
git commit -m "test: rewrite hass tests for HassAgent delegation"
```

---

### Task 2: Implement HassAgent

**Files:**
- Rewrite: `src/prot/hass.py`

**Step 1: Write the implementation**

Replace the entire `src/prot/hass.py` with:

```python
"""Home Assistant integration — conversation API delegation."""

from __future__ import annotations

import httpx

from prot.logging import get_logger

logger = get_logger(__name__)

_HASS_TIMEOUT = httpx.Timeout(15.0)

_TOOL_SCHEMA: dict = {
    "name": "hass_request",
    "description": (
        "Send a natural language command to the Home Assistant agent. "
        "Use for any smart home task: device control, state queries, "
        "automation triggers, scene activation, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Natural language command for Home Assistant. "
                    "Be specific with device names and values. "
                    "Example: '조명 밝기 40%, 색온도 2700K로 변경'"
                ),
            }
        },
        "required": ["command"],
    },
}


class HassAgent:
    """Delegate smart home commands to HA's conversation agent."""

    def __init__(self, url: str, token: str) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._client = httpx.AsyncClient(timeout=_HASS_TIMEOUT)

    async def request(self, command: str) -> str:
        """Send command to HA conversation agent, return response text."""
        try:
            resp = await self._client.post(
                f"{self._url}/api/conversation/process",
                headers={"Authorization": f"Bearer {self._token}"},
                json={"text": command, "language": "ko"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["response"]["speech"]["plain"]["speech"]
        except httpx.ConnectError:
            logger.warning("HASS connection failed")
            return "Home Assistant에 연결할 수 없습니다"
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                logger.warning("HASS auth failed")
                return "Home Assistant 인증 실패"
            logger.warning("HASS HTTP error", status=status)
            return f"Home Assistant 오류 (HTTP {status})"
        except httpx.TimeoutException:
            logger.warning("HASS request timed out")
            return "Home Assistant 응답 시간 초과"

    def build_tool(self) -> dict:
        """Return single tool schema for Claude."""
        return dict(_TOOL_SCHEMA)

    async def close(self) -> None:
        """Close the httpx client."""
        await self._client.aclose()
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_hass.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/prot/hass.py
git commit -m "feat: replace HassRegistry with HassAgent conversation delegation"
```

---

### Task 3: Update context.py — build_tools signature

**Files:**
- Modify: `src/prot/context.py:54-76`
- Modify: `tests/test_context.py:53-89`

**Step 1: Write the failing test updates**

In `tests/test_context.py`, replace the three hass-related tests (lines 53-89):

Replace `test_build_tools_with_hass_registry` (lines 53-63) with:
```python
    def test_build_tools_with_hass_agent(self):
        cm = ContextManager(persona_text="test", rag_context="")
        mock_agent = MagicMock()
        mock_agent.build_tool.return_value = {
            "name": "hass_request",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        }
        tools = cm.build_tools(hass_agent=mock_agent)
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        assert names == ["web_search", "hass_request"]
```

Replace `test_build_tools_with_hass_last_tool_has_cache_control` (lines 70-79) with:
```python
    def test_build_tools_with_hass_last_tool_has_cache_control(self):
        cm = ContextManager(persona_text="test", rag_context="")
        mock_agent = MagicMock()
        mock_agent.build_tool.return_value = {"name": "hass_request", "input_schema": {}}
        tools = cm.build_tools(hass_agent=mock_agent)
        assert "cache_control" in tools[-1]
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}
```

Replace `test_build_tools_adds_cache_control_when_registry_omits_it` (lines 81-89) with:
```python
    def test_build_tools_adds_cache_control_to_hass_tool(self):
        cm = ContextManager(persona_text="test", rag_context="")
        mock_agent = MagicMock()
        mock_agent.build_tool.return_value = {"name": "hass_request", "input_schema": {}}
        tools = cm.build_tools(hass_agent=mock_agent)
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context.py -v -k "hass"`
Expected: FAIL — `build_tools` still expects `hass_registry` kwarg

**Step 3: Update context.py**

In `src/prot/context.py`, replace `build_tools` method (lines 54-76):

Old:
```python
    def build_tools(self, hass_registry=None) -> list[dict]:
        """Build tool definitions with cache on last tool."""
        web_search: dict = {
            ...
        }
        tools = [web_search]

        if hass_registry is not None:
            tools.extend(hass_registry.build_tool_schemas())

        # Ensure last tool has cache_control
        if "cache_control" not in tools[-1]:
            tools[-1]["cache_control"] = {"type": "ephemeral"}

        return tools
```

New:
```python
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_context.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/prot/context.py tests/test_context.py
git commit -m "refactor: update build_tools to accept hass_agent instead of hass_registry"
```

---

### Task 4: Update pipeline.py — HassRegistry → HassAgent

**Files:**
- Modify: `src/prot/pipeline.py` (5 edit sites)

**Step 1: Update pipeline.py**

Make these changes in `src/prot/pipeline.py`:

1. **Line 53** — rename field:
   - Old: `self._hass_registry = None`
   - New: `self._hass_agent = None`

2. **Lines 81-88** — startup() HASS init block. Replace:
   ```python
        try:
            from prot.hass import HassRegistry
            if settings.hass_token:
                self._hass_registry = HassRegistry(settings.hass_url, settings.hass_token)
                await self._hass_registry.discover()
                logger.info("HASS registry ready", entities=len(self._hass_registry._entities))
        except Exception:
            logger.warning("HASS registry not available")
   ```
   With:
   ```python
        try:
            from prot.hass import HassAgent
            if settings.hass_token:
                self._hass_agent = HassAgent(settings.hass_url, settings.hass_token)
                logger.info("HASS agent ready")
        except Exception:
            logger.warning("HASS agent not available")
   ```

3. **Line 255** — build_tools call:
   - Old: `tools = self._ctx.build_tools(hass_registry=self._hass_registry)`
   - New: `tools = self._ctx.build_tools(hass_agent=self._hass_agent)`

4. **Lines 387-388** — tool dispatch. Replace:
   ```python
                        if block.name in ("hass_control", "hass_query") and self._hass_registry:
                            result = await self._hass_registry.execute(block.name, block.input)
   ```
   With:
   ```python
                        if block.name == "hass_request" and self._hass_agent:
                            result = await self._hass_agent.request(block.input["command"])
   ```

5. **Lines 569-570** — shutdown closeables. Replace:
   - Old: `if self._hass_registry is not None:` / `closeables.append(self._hass_registry.close)`
   - New: `if self._hass_agent is not None:` / `closeables.append(self._hass_agent.close)`

**Step 2: Verify no import errors**

Run: `uv run python -c "from prot.pipeline import Pipeline; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/prot/pipeline.py
git commit -m "refactor: replace HassRegistry with HassAgent in pipeline"
```

---

### Task 5: Update pipeline tests

**Files:**
- Modify: `tests/test_pipeline.py`

**Step 1: Update test references**

Global replacements in `tests/test_pipeline.py`:

1. In `_make_pipeline()` (line 73):
   - `p._hass_registry = None` → `p._hass_agent = None`

2. In all tool loop tests that use `hass_control` (lines 797, 852, 937, 992):
   - `tool_block.name = "hass_control"` → `tool_block.name = "hass_request"`
   - `tool_block.input = {"entity_id": ..., "action": ...}` → `tool_block.input = {"command": "test command"}`

3. In all tests that set `_hass_registry` (lines 812, 867, 953, 1006):
   - `p._hass_registry = mock_registry` → `p._hass_agent = mock_agent`
   - `mock_registry = AsyncMock()` / `mock_registry.execute = AsyncMock(...)` → `mock_agent = AsyncMock()` / `mock_agent.request = AsyncMock(return_value="done")`

4. Replace `TestPipelineHassRouting` class (lines 1170-1247):

```python
class TestPipelineHassRouting:
    async def test_hass_tool_routed_to_agent(self):
        """hass_request tool calls are routed through _hass_agent."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value="조명을 켰습니다")
        p._hass_agent = mock_agent

        call_count = 0

        async def fake_stream(*a, **kw):
            nonlocal call_count
            call_count += 1
            yield "Checking." if call_count == 1 else "Done."

        p._llm.stream_response = fake_stream

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "hass_request"
        tool_block.id = "tool_hass"
        tool_block.input = {"command": "거실 조명 켜줘"}

        tc = 0

        def fake_get_tool():
            nonlocal tc
            tc += 1
            return [tool_block] if tc == 1 else []

        p._llm.get_tool_use_blocks = fake_get_tool
        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        mock_agent.request.assert_awaited_once_with("거실 조명 켜줘")

    async def test_build_tools_called_with_agent(self):
        """build_tools receives hass_agent parameter."""
        p = _make_pipeline()
        p._sm.on_speech_detected()
        p._sm.on_utterance_complete()

        mock_agent = MagicMock()
        p._hass_agent = mock_agent

        async def fake_stream(*a, **kw):
            yield "Hello."

        p._llm.stream_response = fake_stream
        p._llm.last_response_content = [MagicMock(type="text")]

        async def fake_tts(text):
            yield b"\x00" * 100

        p._tts.stream_audio = fake_tts

        with patch("prot.pipeline.settings") as ms:
            ms.claude_model = "test"
            ms.active_timeout = 999
            ms.tts_sentence_silence_ms = 0
            ms.elevenlabs_output_format = "pcm_24000"
            await p._process_response()

        p._ctx.build_tools.assert_called_with(hass_agent=mock_agent)
```

**Step 2: Run all pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: update pipeline tests for HassAgent"
```

---

### Task 6: Run full test suite and verify

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 2: Verify no stale imports**

Run: `uv run python -c "from prot.hass import HassAgent; print('OK')"`
Run: `uv run python -c "from prot.pipeline import Pipeline; print('OK')"`
Run: `uv run python -c "from prot.context import ContextManager; print('OK')"`
Expected: All print `OK`

**Step 3: Check for stale references in codebase**

Run: `grep -r "HassRegistry\|hass_control\|hass_query\|parse_color" src/ tests/`
Expected: No matches

**Step 4: Commit if any fixes needed**

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update architecture description**

In CLAUDE.md, update the `hass.py` line in the Architecture section:
- Old: `hass.py           # Home Assistant API client (get_state, call_service)`
- New: `hass.py           # Home Assistant conversation API delegation (HassAgent)`

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for HassAgent delegation"
```
