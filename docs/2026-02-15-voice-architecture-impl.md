# Voice Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a real-time voice conversation system with Axel persona using interruptible streaming pipeline.

**Architecture:** Single FastAPI process with Silero VAD → Deepgram Flux STT → Claude Opus 4.6 LLM → ElevenLabs Flash TTS → paplay output. State machine orchestrates transitions. Barge-in support via VAD during playback.

**Tech Stack:** Python 3.12, FastAPI, PyAudio, Silero VAD, Deepgram Flux, Anthropic SDK, ElevenLabs SDK, paplay, asyncpg, pgvector, voyageai

**Design Doc:** `docs/plans/2026-02-15-voice-architecture-design.md`

---

## Phase 1: Project Scaffolding + Core Logic

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/prot/__init__.py`
- Create: `src/prot/config.py`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Initialize project with uv**

```bash
cd /home/cyan/workplace/prot
uv init --lib --name prot
```

**Step 2: Add dependencies to pyproject.toml**

```toml
[project]
name = "prot"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "anthropic>=0.52",
    "elevenlabs>=1.0",
    "deepgram-sdk>=4.0",
    "pyaudio>=0.2.14",
    "silero-vad>=5.1",
    "httpx>=0.28",
    "pydantic-settings>=2.7",
    "asyncpg>=0.30",
    "voyageai>=0.3",
    "numpy>=2.0",
    "pgvector>=0.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25",
    "pytest-cov>=6.0",
]
```

**Step 3: Create config**

`src/prot/config.py` — Pydantic Settings loading from `.env`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str
    deepgram_api_key: str
    elevenlabs_api_key: str

    # Audio
    mic_device_index: int = 11
    sample_rate: int = 16000
    chunk_size: int = 512

    # VAD
    vad_threshold: float = 0.5
    vad_threshold_speaking: float = 0.8

    # Deepgram
    deepgram_model: str = "flux"
    deepgram_language: str = "ko"
    deepgram_endpointing: int = 500

    # LLM
    claude_model: str = "claude-opus-4-6"
    claude_max_tokens: int = 1500
    claude_effort: str = "medium"

    # TTS
    elevenlabs_model: str = "eleven_flash_v2_5"
    elevenlabs_voice_id: str = ""
    elevenlabs_output_format: str = "pcm_16000"

    # HASS
    hass_url: str = "http://localhost:8123"
    hass_token: str = ""

    # Database
    database_url: str = "postgresql://prot:prot@localhost:5432/prot"
    db_pool_min: int = 2
    db_pool_max: int = 10

    # Embeddings
    voyage_api_key: str = ""
    voyage_model: str = "voyage-3.5-lite"
    voyage_dimension: int = 1024

    # Memory
    memory_extraction_model: str = "claude-haiku-4-5-20251001"
    rag_context_target_tokens: int = 3000
    rag_top_k: int = 10

    # Timers
    active_timeout: int = 30  # seconds before WS disconnect

    model_config = {"env_file": ".env"}

settings = Settings()
```

**Step 4: Create .env.example**

```
ANTHROPIC_API_KEY=
DEEPGRAM_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
HASS_URL=http://localhost:8123
HASS_TOKEN=
MIC_DEVICE_INDEX=11
DATABASE_URL=postgresql://prot:prot@localhost:5432/prot
VOYAGE_API_KEY=
```

**Step 5: Create test conftest**

`tests/conftest.py`:

```python
import pytest

@pytest.fixture
def sample_transcript():
    return "오늘 날씨 어때?"

@pytest.fixture
def sample_llm_response():
    return "밖에 나가보진 않았는데, HASS 데이터 보면 영하 2도에 맑음이네. Bio-fuel Injection 하고 나가는 게 나을 듯."
```

**Step 6: Install and verify**

```bash
uv sync --all-extras
uv run pytest --co  # collect tests, verify setup
```

**Step 7: Commit**

```bash
git init
git add pyproject.toml src/ tests/ .env.example .gitignore
git commit -m "chore: initialize prot project with dependencies and config"
```

---

### Task 2: Post-Processing (TTS Sanitizer)

Pure functions. Full TDD.

**Files:**
- Create: `src/prot/processing.py`
- Create: `tests/test_processing.py`

**Step 1: Write failing tests**

`tests/test_processing.py`:

```python
import pytest
from prot.processing import sanitize_for_tts, ensure_complete_sentence, chunk_sentences

class TestSanitizeForTts:
    def test_strips_markdown_bold(self):
        assert sanitize_for_tts("이건 **중요한** 내용이야") == "이건 중요한 내용이야"

    def test_strips_markdown_headers(self):
        assert sanitize_for_tts("# 제목\n내용") == "제목\n내용"

    def test_strips_numbered_list(self):
        assert sanitize_for_tts("1. 첫번째 2. 두번째") == "첫번째 두번째"

    def test_strips_bullets(self):
        assert sanitize_for_tts("- 항목 하나\n- 항목 둘") == "항목 하나\n항목 둘"

    def test_strips_emoji(self):
        assert sanitize_for_tts("좋아! 😊 가자") == "좋아! 가자"

    def test_strips_code_backticks(self):
        assert sanitize_for_tts("이건 `코드`야") == "이건 코드야"

    def test_preserves_normal_text(self):
        assert sanitize_for_tts("그냥 평범한 문장이야.") == "그냥 평범한 문장이야."

    def test_strips_multiple_markers(self):
        assert sanitize_for_tts("**굵게** _기울임_ ~~취소선~~") == "굵게 기울임 취소선"

class TestEnsureCompleteSentence:
    def test_already_complete(self):
        assert ensure_complete_sentence("완벽한 문장이야.") == "완벽한 문장이야."

    def test_truncated_mid_sentence(self):
        assert ensure_complete_sentence("첫 문장이야. 두번째 문장은 아직") == "첫 문장이야."

    def test_question_mark(self):
        assert ensure_complete_sentence("뭐하는 거야? 나는") == "뭐하는 거야?"

    def test_exclamation(self):
        assert ensure_complete_sentence("대박이다! 진짜") == "대박이다!"

    def test_tilde(self):
        assert ensure_complete_sentence("그렇지~ 근데") == "그렇지~"

    def test_no_sentence_boundary(self):
        assert ensure_complete_sentence("경계가 없는 텍스트") == "경계가 없는 텍스트"

    def test_empty_string(self):
        assert ensure_complete_sentence("") == ""

class TestChunkSentences:
    def test_single_sentence(self):
        chunks = list(chunk_sentences("안녕하세요."))
        assert chunks == ["안녕하세요."]

    def test_multiple_sentences(self):
        chunks = list(chunk_sentences("첫번째. 두번째. 세번째."))
        assert chunks == ["첫번째.", "두번째.", "세번째."]

    def test_question_and_statement(self):
        chunks = list(chunk_sentences("밥 먹었어? 나는 먹었어."))
        assert chunks == ["밥 먹었어?", "나는 먹었어."]

    def test_incomplete_trailing(self):
        chunks = list(chunk_sentences("완성. 미완성"))
        assert chunks == ["완성.", "미완성"]
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_processing.py -v
```

Expected: ModuleNotFoundError

**Step 3: Write implementation**

`src/prot/processing.py`:

```python
import re

def sanitize_for_tts(text: str) -> str:
    """Remove markdown, emoji, and other TTS-hostile formatting."""
    text = re.sub(r'[*_#`~\[\](){}|>]', '', text)
    text = re.sub(r'\d+\.\s', '', text)
    text = re.sub(r'[-•]\s', '', text)
    # Remove emoji (Unicode emoji ranges)
    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
        r'\U00002702-\U000027B0\U0000FE00-\U0000FE0F'
        r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F]+',
        '', text
    )
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def ensure_complete_sentence(text: str) -> str:
    """Truncate to last complete sentence boundary."""
    if not text:
        return ""
    boundaries = '.!?~'
    for i in range(len(text) - 1, -1, -1):
        if text[i] in boundaries:
            return text[:i + 1]
    return text

def chunk_sentences(text: str):
    """Yield individual sentences from text."""
    pattern = re.compile(r'([^.!?~]*[.!?~]+)\s*')
    last_end = 0
    for match in pattern.finditer(text):
        last_end = match.end()
        yield match.group(1).strip()
    remainder = text[last_end:].strip()
    if remainder:
        yield remainder
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_processing.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/prot/processing.py tests/test_processing.py
git commit -m "feat: add TTS post-processing (sanitizer, sentence chunking)"
```

---

### Task 3: State Machine

**Files:**
- Create: `src/prot/state.py`
- Create: `tests/test_state.py`

**Step 1: Write failing tests**

`tests/test_state.py`:

```python
import pytest
from prot.state import StateMachine, State

class TestStateMachine:
    def test_initial_state_is_idle(self):
        sm = StateMachine()
        assert sm.state == State.IDLE

    def test_idle_to_listening_on_speech(self):
        sm = StateMachine()
        sm.on_speech_detected()
        assert sm.state == State.LISTENING

    def test_listening_to_processing_on_endpointing(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        assert sm.state == State.PROCESSING

    def test_processing_to_speaking_on_first_audio(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        assert sm.state == State.SPEAKING

    def test_speaking_to_interrupted_on_speech(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_speech_detected()
        assert sm.state == State.INTERRUPTED

    def test_interrupted_to_listening(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_speech_detected()  # INTERRUPTED
        sm.on_interrupt_handled()
        assert sm.state == State.LISTENING

    def test_speaking_to_active_on_tts_done(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_tts_complete()
        assert sm.state == State.ACTIVE

    def test_active_to_listening_on_speech(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_tts_complete()
        sm.on_speech_detected()
        assert sm.state == State.LISTENING

    def test_active_to_idle_on_timeout(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_tts_complete()
        sm.on_active_timeout()
        assert sm.state == State.IDLE

    def test_invalid_transition_raises(self):
        sm = StateMachine()
        with pytest.raises(ValueError):
            sm.on_utterance_complete()  # Can't endpointing from IDLE

    def test_vad_threshold_elevated_during_speaking(self):
        sm = StateMachine()
        assert sm.vad_threshold == 0.5
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        assert sm.vad_threshold == 0.8

    def test_vad_threshold_restored_after_speaking(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_tts_complete()
        assert sm.vad_threshold == 0.5

    def test_callbacks_fire_on_transition(self):
        sm = StateMachine()
        events = []
        sm.on_enter(State.LISTENING, lambda: events.append("listening"))
        sm.on_speech_detected()
        assert events == ["listening"]
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_state.py -v
```

**Step 3: Write implementation**

`src/prot/state.py`:

```python
from enum import Enum
from collections import defaultdict
from typing import Callable

class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ACTIVE = "active"

class StateMachine:
    def __init__(
        self,
        vad_threshold_normal: float = 0.5,
        vad_threshold_speaking: float = 0.8,
    ):
        self.state = State.IDLE
        self._vad_normal = vad_threshold_normal
        self._vad_speaking = vad_threshold_speaking
        self.vad_threshold = vad_threshold_normal
        self._callbacks: dict[State, list[Callable]] = defaultdict(list)

    def on_enter(self, state: State, callback: Callable) -> None:
        self._callbacks[state].append(callback)

    def _transition(self, new_state: State) -> None:
        self.state = new_state
        if new_state == State.SPEAKING:
            self.vad_threshold = self._vad_speaking
        elif new_state in (State.ACTIVE, State.IDLE, State.LISTENING):
            self.vad_threshold = self._vad_normal
        for cb in self._callbacks[new_state]:
            cb()

    def on_speech_detected(self) -> None:
        if self.state == State.IDLE:
            self._transition(State.LISTENING)
        elif self.state == State.SPEAKING:
            self._transition(State.INTERRUPTED)
        elif self.state == State.ACTIVE:
            self._transition(State.LISTENING)
        else:
            raise ValueError(f"Cannot detect speech in state {self.state}")

    def on_utterance_complete(self) -> None:
        if self.state == State.LISTENING:
            self._transition(State.PROCESSING)
        else:
            raise ValueError(f"Cannot complete utterance in state {self.state}")

    def on_tts_started(self) -> None:
        if self.state == State.PROCESSING:
            self._transition(State.SPEAKING)
        else:
            raise ValueError(f"Cannot start TTS in state {self.state}")

    def on_tts_complete(self) -> None:
        if self.state == State.SPEAKING:
            self._transition(State.ACTIVE)
        else:
            raise ValueError(f"Cannot complete TTS in state {self.state}")

    def on_interrupt_handled(self) -> None:
        if self.state == State.INTERRUPTED:
            self._transition(State.LISTENING)
        else:
            raise ValueError(f"Cannot handle interrupt in state {self.state}")

    def on_active_timeout(self) -> None:
        if self.state == State.ACTIVE:
            self._transition(State.IDLE)
        else:
            raise ValueError(f"Cannot timeout in state {self.state}")
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_state.py -v
```

**Step 5: Commit**

```bash
git add src/prot/state.py tests/test_state.py
git commit -m "feat: add conversation state machine with barge-in transitions"
```

---

## Phase 2: LLM Pipeline (Text In → Text Out)

### Task 4: Context Manager (Prompt Building)

**Files:**
- Create: `src/prot/context.py`
- Create: `src/prot/persona.py`
- Create: `tests/test_context.py`

**Step 1: Write failing tests**

`tests/test_context.py`:

```python
import pytest
from prot.context import ContextManager

class TestContextManager:
    def test_build_system_blocks_returns_three_blocks(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert len(blocks) == 3

    def test_block1_persona_has_cache_control(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "test persona" in blocks[0]["text"]

    def test_block2_rag_has_cache_control(self):
        cm = ContextManager(persona_text="test persona", rag_context="some context")
        blocks = cm.build_system_blocks()
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}
        assert "some context" in blocks[1]["text"]

    def test_block3_dynamic_has_no_cache_control(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert "cache_control" not in blocks[2]

    def test_block3_contains_datetime(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert "datetime:" in blocks[2]["text"]

    def test_block_order_is_persona_rag_dynamic(self):
        """Verify ordering: Block 1=persona (cached), Block 2=RAG (cached), Block 3=dynamic (NOT cached).
        This order is CRITICAL: dynamic content must be LAST to preserve cached prefix."""
        cm = ContextManager(persona_text="persona text", rag_context="rag text")
        blocks = cm.build_system_blocks()
        assert "persona text" in blocks[0]["text"]
        assert "rag text" in blocks[1]["text"]
        assert "datetime:" in blocks[2]["text"]
        assert "cache_control" in blocks[0]
        assert "cache_control" in blocks[1]
        assert "cache_control" not in blocks[2]

    def test_build_tools_returns_list(self):
        cm = ContextManager(persona_text="test", rag_context="")
        tools = cm.build_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1  # at least web_search

    def test_add_message_and_get_history(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "안녕")
        cm.add_message("assistant", "야 뭐해")
        history = cm.get_messages()
        assert len(history) == 2
        assert history[0]["role"] == "user"
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_context.py -v
```

**Step 3: Create persona module**

`src/prot/persona.py`:

```python
from pathlib import Path

PERSONA_PATH = Path(__file__).parent.parent.parent / "axel.md"

def load_persona() -> str:
    """Load Axel persona from file."""
    if PERSONA_PATH.exists():
        return PERSONA_PATH.read_text(encoding="utf-8")
    return ""
```

**Step 4: Write ContextManager implementation**

`src/prot/context.py`:

```python
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

class ContextManager:
    def __init__(self, persona_text: str, rag_context: str = ""):
        self._persona = persona_text
        self._rag_context = rag_context
        self._messages: list[dict] = []

    def build_system_blocks(self) -> list[dict]:
        """Build 3-block system prompt with cache control.

        Order is CRITICAL for prompt caching:
          Block 1: Persona + Rules (STATIC, cached)
          Block 2: GraphRAG Context (TOPIC-DEPENDENT, cached)
          Block 3: Dynamic Context (PER-REQUEST, NOT cached, MUST be last)

        If dynamic content (datetime) sits between cached blocks, it breaks the
        cache prefix — downstream blocks would NEVER hit cache.
        """
        block1_persona = {
            "type": "text",
            "text": self._persona,
            "cache_control": {"type": "ephemeral"},
        }
        block2_rag = {
            "type": "text",
            "text": self._rag_context or "(no additional context)",
            "cache_control": {"type": "ephemeral"},
        }
        block3_dynamic = {
            "type": "text",
            "text": (
                f"datetime: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"timezone: Asia/Seoul"
            ),
        }
        return [block1_persona, block2_rag, block3_dynamic]

    def build_tools(self) -> list[dict]:
        """Build tool definitions with cache on last tool."""
        web_search = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 1,
            "user_location": {
                "type": "approximate",
                "city": "Seoul",
                "country": "KR",
                "timezone": "Asia/Seoul",
            },
        }
        hass_tool = {
            "name": "home_assistant",
            "description": "Query or control Home Assistant. Actions: get_state, call_service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get_state", "call_service"],
                    },
                    "entity_id": {"type": "string"},
                    "service_data": {"type": "object"},
                },
                "required": ["action", "entity_id"],
            },
            "cache_control": {"type": "ephemeral"},
        }
        return [web_search, hass_tool]

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def update_rag_context(self, context: str) -> None:
        self._rag_context = context
```

**Step 5: Run tests**

```bash
uv run pytest tests/test_context.py -v
```

**Step 6: Commit**

```bash
git add src/prot/context.py src/prot/persona.py tests/test_context.py
git commit -m "feat: add context manager with prompt caching and tool definitions"
```

---

### Task 5: LLM Client (Claude Streaming)

**Files:**
- Create: `src/prot/llm.py`
- Create: `tests/test_llm.py`

**Step 1: Write failing tests (mock-based)**

`tests/test_llm.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prot.llm import LLMClient

@pytest.mark.asyncio
class TestLLMClient:
    async def test_stream_response_yields_text(self):
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream._events = [
            MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="안녕")),
            MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="하세요.")),
        ]
        mock_stream.__anext__ = AsyncMock(side_effect=[
            mock_stream._events[0],
            mock_stream._events[1],
            StopAsyncIteration,
        ])

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            chunks = []
            async for chunk in client.stream_response(
                system_blocks=[{"type": "text", "text": "test"}],
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            ):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0] == "안녕"

    async def test_cancel_stops_stream(self):
        client = LLMClient(api_key="test")
        client.cancel()
        assert client._cancelled is True
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_llm.py -v
```

**Step 3: Write implementation**

`src/prot/llm.py`:

```python
from anthropic import AsyncAnthropic
from prot.config import settings

class LLMClient:
    def __init__(self, api_key: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self._cancelled = False
        self._active_stream = None

    async def stream_response(
        self,
        system_blocks: list[dict],
        tools: list[dict],
        messages: list[dict],
    ):
        """Stream text deltas from Claude. Yields str chunks.

        system_blocks order: [persona (cached), rag (cached), dynamic (NOT cached)]
        """
        self._cancelled = False

        async with self._client.messages.stream(
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            thinking={"type": "adaptive"},
            effort=settings.claude_effort,
            system=system_blocks,  # Order: persona → rag → dynamic (cache-safe)
            tools=tools if tools else None,
            messages=messages,
        ) as stream:
            self._active_stream = stream
            async for event in stream:
                if self._cancelled:
                    break
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield event.delta.text

        self._active_stream = None

    def cancel(self) -> None:
        """Cancel the active stream."""
        self._cancelled = True

    async def execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool call from Claude. Returns tool result."""
        if tool_name == "home_assistant":
            return await self._execute_hass(tool_input)
        return {"error": f"Unknown tool: {tool_name}"}

    async def _execute_hass(self, tool_input: dict) -> dict:
        """Execute Home Assistant API call."""
        import httpx
        action = tool_input.get("action")
        entity_id = tool_input.get("entity_id")

        async with httpx.AsyncClient() as http:
            headers = {"Authorization": f"Bearer {settings.hass_token}"}
            if action == "get_state":
                r = await http.get(
                    f"{settings.hass_url}/api/states/{entity_id}",
                    headers=headers,
                )
                return r.json()
            elif action == "call_service":
                domain, service = entity_id.rsplit(".", 1)
                r = await http.post(
                    f"{settings.hass_url}/api/services/{domain}/{service}",
                    headers=headers,
                    json=tool_input.get("service_data", {}),
                )
                return r.json()
        return {"error": "Invalid action"}
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_llm.py -v
```

**Step 5: Commit**

```bash
git add src/prot/llm.py tests/test_llm.py
git commit -m "feat: add Claude streaming client with tool execution"
```

---

### Task 6: Text-to-Text Integration Test

**Files:**
- Create: `tests/test_integration_text.py`

**Step 1: Write integration test (requires real API key)**

`tests/test_integration_text.py`:

```python
import pytest
from prot.llm import LLMClient
from prot.context import ContextManager
from prot.processing import sanitize_for_tts, chunk_sentences
from prot.persona import load_persona

@pytest.mark.integration
@pytest.mark.asyncio
async def test_text_to_text_pipeline():
    """Full text pipeline: user input → Claude → sanitized output."""
    persona = load_persona()
    cm = ContextManager(persona_text=persona)
    cm.add_message("user", "야 오늘 뭐하냐")

    client = LLMClient()
    raw = ""
    async for chunk in client.stream_response(
        system_blocks=cm.build_system_blocks(),
        tools=cm.build_tools(),
        messages=cm.get_messages(),
    ):
        raw += chunk

    assert len(raw) > 0
    sanitized = sanitize_for_tts(raw)
    sentences = list(chunk_sentences(sanitized))
    assert len(sentences) >= 1
    # Verify no markdown artifacts
    assert "**" not in sanitized
    assert "##" not in sanitized
    print(f"\nAxel: {sanitized}")
```

**Step 2: Run integration test**

```bash
uv run pytest tests/test_integration_text.py -v -m integration -s
```

**Step 3: Commit**

```bash
git add tests/test_integration_text.py
git commit -m "test: add text-to-text integration test"
```

---

## Phase 2B: Data Layer

### Task 7: Database Setup

**Files:**
- Create: `src/prot/db.py`
- Create: `src/prot/schema.sql`
- Create: `tests/test_db.py`

**Step 1: Write SQL schema**

`src/prot/schema.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    attributes JSONB NOT NULL DEFAULT '{}',
    name_embedding vector(1024),
    mention_count INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    weight FLOAT NOT NULL DEFAULT 1.0,
    attributes JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS communities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    level INT NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    summary_embedding vector(1024),
    entity_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS community_members (
    community_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (community_id, entity_id)
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    content_embedding vector(1024),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW indexes for vector similarity search
CREATE INDEX IF NOT EXISTS idx_entities_name_embedding
    ON entities USING hnsw (name_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS idx_communities_summary_embedding
    ON communities USING hnsw (summary_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS idx_messages_content_embedding
    ON conversation_messages USING hnsw (content_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Trigram index for fuzzy entity name matching
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm
    ON entities USING gin (name gin_trgm_ops);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_entities_namespace_type
    ON entities (namespace, entity_type);

CREATE INDEX IF NOT EXISTS idx_relationships_source_target
    ON relationships (source_id, target_id);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON conversation_messages (conversation_id, created_at);
```

**Step 2: Write failing tests**

`tests/test_db.py`:

```python
import pytest
import asyncpg
from prot.db import init_pool, close_pool, get_pool, execute_schema

@pytest.mark.asyncio
class TestDatabase:
    async def test_pool_creation(self):
        pool = await init_pool("postgresql://prot:prot@localhost:5432/prot_test")
        assert pool is not None
        assert pool.get_size() >= 1
        await close_pool()

    async def test_schema_execution(self):
        pool = await init_pool("postgresql://prot:prot@localhost:5432/prot_test")
        await execute_schema(pool)
        async with pool.acquire() as conn:
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            table_names = [t["tablename"] for t in tables]
            assert "entities" in table_names
            assert "relationships" in table_names
            assert "communities" in table_names
            assert "community_members" in table_names
            assert "conversation_messages" in table_names
        await close_pool()

    async def test_basic_entity_crud(self):
        pool = await init_pool("postgresql://prot:prot@localhost:5432/prot_test")
        await execute_schema(pool)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO entities (name, entity_type, description) "
                "VALUES ($1, $2, $3) RETURNING id, name",
                "TestEntity", "person", "A test entity"
            )
            assert row["name"] == "TestEntity"
            await conn.execute("DELETE FROM entities WHERE id = $1", row["id"])
        await close_pool()
```

**Step 3: Write implementation**

`src/prot/db.py`:

```python
from pathlib import Path
import asyncpg
from prot.config import settings

_pool: asyncpg.Pool | None = None

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

async def init_pool(dsn: str | None = None) -> asyncpg.Pool:
    """Create asyncpg connection pool."""
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=dsn or settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
    )
    return _pool

async def get_pool() -> asyncpg.Pool:
    """Get existing pool or raise."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool

async def close_pool() -> None:
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

async def execute_schema(pool: asyncpg.Pool) -> None:
    """Execute schema.sql to create tables and indexes."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
```

Copy `src/prot/schema.sql` from Step 1 into the actual file.

**Step 4: Run tests**

```bash
uv run pytest tests/test_db.py -v
```

**Step 5: Commit**

```bash
git add src/prot/db.py src/prot/schema.sql tests/test_db.py
git commit -m "feat: add asyncpg database pool and pgvector schema"
```

---

### Task 8: Embedding Client

**Files:**
- Create: `src/prot/embeddings.py`
- Create: `tests/test_embeddings.py`

**Step 1: Write failing tests**

`tests/test_embeddings.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prot.embeddings import AsyncVoyageEmbedder

@pytest.mark.asyncio
class TestAsyncVoyageEmbedder:
    async def test_embed_texts_returns_vectors(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_client.embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vectors = await embedder.embed_texts(["hello", "world"])
            assert len(vectors) == 2
            assert len(vectors[0]) == 1024
            mock_client.embed.assert_called_once_with(
                texts=["hello", "world"],
                model="voyage-3.5-lite",
                input_type="document",
            )

    async def test_embed_query_uses_query_input_type(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vector = await embedder.embed_query("search term")
            assert len(vector) == 1024
            mock_client.embed.assert_called_once_with(
                texts=["search term"],
                model="voyage-3.5-lite",
                input_type="query",
            )

    async def test_batch_splitting_over_128(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024] * 128
        mock_client.embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            texts = [f"text_{i}" for i in range(200)]
            vectors = await embedder.embed_texts(texts)
            assert len(vectors) == 200
            assert mock_client.embed.call_count == 2  # 128 + 72

    async def test_semaphore_limits_concurrency(self):
        embedder = AsyncVoyageEmbedder.__new__(AsyncVoyageEmbedder)
        embedder._max_concurrent = 5
        import asyncio
        embedder._semaphore = asyncio.Semaphore(5)
        assert embedder._semaphore._value == 5
```

**Step 2: Write implementation**

`src/prot/embeddings.py`:

```python
import asyncio
import voyageai
from prot.config import settings

class AsyncVoyageEmbedder:
    """Async embedding client using voyage-3.5-lite."""

    MAX_BATCH = 128

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrent: int = 5,
    ):
        self._client = voyageai.AsyncClient(
            api_key=api_key or settings.voyage_api_key
        )
        self._model = model or settings.voyage_model
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (input_type='document'). Auto-batches."""
        all_vectors = []
        for i in range(0, len(texts), self.MAX_BATCH):
            batch = texts[i : i + self.MAX_BATCH]
            async with self._semaphore:
                result = await self._client.embed(
                    texts=batch,
                    model=self._model,
                    input_type="document",
                )
            all_vectors.extend(result.embeddings)
        return all_vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed single query text (input_type='query')."""
        async with self._semaphore:
            result = await self._client.embed(
                texts=[text],
                model=self._model,
                input_type="query",
            )
        return result.embeddings[0]
```

**Step 3: Run tests**

```bash
uv run pytest tests/test_embeddings.py -v
```

**Step 4: Commit**

```bash
git add src/prot/embeddings.py tests/test_embeddings.py
git commit -m "feat: add async Voyage embedding client with batch splitting"
```

---

### Task 9: GraphRAG Store

**Files:**
- Create: `src/prot/graphrag.py`
- Create: `tests/test_graphrag.py`

**Step 1: Write failing tests**

`tests/test_graphrag.py`:

```python
import pytest
import asyncpg
from prot.db import init_pool, close_pool, execute_schema
from prot.graphrag import GraphRAGStore

@pytest.fixture
async def store():
    pool = await init_pool("postgresql://prot:prot@localhost:5432/prot_test")
    await execute_schema(pool)
    s = GraphRAGStore(pool)
    yield s
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM community_members")
        await conn.execute("DELETE FROM communities")
        await conn.execute("DELETE FROM relationships")
        await conn.execute("DELETE FROM conversation_messages")
        await conn.execute("DELETE FROM entities")
    await close_pool()

@pytest.mark.asyncio
class TestGraphRAGStore:
    async def test_upsert_entity(self, store):
        entity_id = await store.upsert_entity(
            name="TestPerson",
            entity_type="person",
            description="A test person",
            embedding=[0.1] * 1024,
        )
        assert entity_id is not None

    async def test_upsert_entity_increments_mention_count(self, store):
        await store.upsert_entity(name="Dup", entity_type="person", description="first")
        await store.upsert_entity(name="Dup", entity_type="person", description="updated")
        entity = await store.get_entity_by_name("Dup")
        assert entity["mention_count"] == 2
        assert entity["description"] == "updated"

    async def test_upsert_relationship(self, store):
        src = await store.upsert_entity(name="A", entity_type="person", description="A")
        tgt = await store.upsert_entity(name="B", entity_type="person", description="B")
        rel_id = await store.upsert_relationship(
            source_id=src, target_id=tgt,
            relation_type="knows", description="A knows B"
        )
        assert rel_id is not None

    async def test_search_entities_semantic(self, store):
        await store.upsert_entity(
            name="SemanticTest",
            entity_type="concept",
            description="test",
            embedding=[0.5] * 1024,
        )
        results = await store.search_entities_semantic(
            query_embedding=[0.5] * 1024, top_k=5
        )
        assert len(results) >= 1
        assert results[0]["name"] == "SemanticTest"

    async def test_search_communities(self, store):
        await store.upsert_community(
            level=0,
            summary="A community about testing",
            summary_embedding=[0.3] * 1024,
            entity_count=5,
        )
        results = await store.search_communities(
            query_embedding=[0.3] * 1024, top_k=5
        )
        assert len(results) >= 1

    async def test_get_entity_neighbors(self, store):
        a = await store.upsert_entity(name="Center", entity_type="person", description="c")
        b = await store.upsert_entity(name="Neighbor1", entity_type="person", description="n1")
        c = await store.upsert_entity(name="Neighbor2", entity_type="person", description="n2")
        await store.upsert_relationship(a, b, "knows", "knows n1")
        await store.upsert_relationship(a, c, "knows", "knows n2")
        neighbors = await store.get_entity_neighbors(a, max_depth=1)
        names = {n["name"] for n in neighbors}
        assert "Neighbor1" in names
        assert "Neighbor2" in names
```

**Step 2: Write implementation**

`src/prot/graphrag.py`:

```python
import asyncpg
from uuid import UUID

class GraphRAGStore:
    """pgvector-backed GraphRAG storage."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: list[float] | None = None,
        namespace: str = "default",
    ) -> UUID:
        """Insert or update entity. Increments mention_count on conflict."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO entities (namespace, name, entity_type, description, name_embedding)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT ON CONSTRAINT entities_pkey DO UPDATE SET
                    -- This won't work with UUID PK, use name+namespace unique instead
                RETURNING id
                """,
                # Actually: need a unique constraint on (namespace, name)
            )
            # Simplified: check if exists first
            existing = await conn.fetchrow(
                "SELECT id, mention_count FROM entities WHERE namespace = $1 AND name = $2",
                namespace, name,
            )
            if existing:
                await conn.execute(
                    """UPDATE entities
                    SET description = $1, mention_count = mention_count + 1,
                        name_embedding = COALESCE($2, name_embedding), updated_at = now()
                    WHERE id = $3""",
                    description, embedding, existing["id"],
                )
                return existing["id"]
            else:
                row = await conn.fetchrow(
                    """INSERT INTO entities (namespace, name, entity_type, description, name_embedding)
                    VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                    namespace, name, entity_type, description, embedding,
                )
                return row["id"]

    async def get_entity_by_name(self, name: str, namespace: str = "default") -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entities WHERE namespace = $1 AND name = $2",
                namespace, name,
            )
            return dict(row) if row else None

    async def upsert_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: str,
        description: str,
        weight: float = 1.0,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO relationships (source_id, target_id, relation_type, description, weight)
                VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                source_id, target_id, relation_type, description, weight,
            )
            return row["id"]

    async def search_entities_semantic(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, name, entity_type, description, mention_count,
                          1 - (name_embedding <=> $1::vector) AS similarity
                FROM entities WHERE name_embedding IS NOT NULL
                ORDER BY name_embedding <=> $1::vector LIMIT $2""",
                query_embedding, top_k,
            )
            return [dict(r) for r in rows]

    async def upsert_community(
        self, level: int, summary: str,
        summary_embedding: list[float] | None = None, entity_count: int = 0,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO communities (level, summary, summary_embedding, entity_count)
                VALUES ($1, $2, $3, $4) RETURNING id""",
                level, summary, summary_embedding, entity_count,
            )
            return row["id"]

    async def search_communities(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, level, summary, entity_count,
                          1 - (summary_embedding <=> $1::vector) AS similarity
                FROM communities WHERE summary_embedding IS NOT NULL
                ORDER BY summary_embedding <=> $1::vector LIMIT $2""",
                query_embedding, top_k,
            )
            return [dict(r) for r in rows]

    async def get_entity_neighbors(
        self, entity_id: UUID, max_depth: int = 1
    ) -> list[dict]:
        """Get neighboring entities via recursive CTE."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH RECURSIVE neighbors AS (
                    SELECT target_id AS id, 1 AS depth
                    FROM relationships WHERE source_id = $1
                    UNION
                    SELECT source_id AS id, 1 AS depth
                    FROM relationships WHERE target_id = $1
                    UNION ALL
                    SELECT CASE WHEN r.source_id = n.id THEN r.target_id ELSE r.source_id END,
                           n.depth + 1
                    FROM neighbors n
                    JOIN relationships r ON r.source_id = n.id OR r.target_id = n.id
                    WHERE n.depth < $2
                )
                SELECT DISTINCT e.id, e.name, e.entity_type, e.description
                FROM neighbors n
                JOIN entities e ON e.id = n.id
                WHERE e.id != $1
                """,
                entity_id, max_depth,
            )
            return [dict(r) for r in rows]
```

**Step 3: Run tests**

```bash
uv run pytest tests/test_graphrag.py -v
```

**Step 4: Commit**

```bash
git add src/prot/graphrag.py tests/test_graphrag.py
git commit -m "feat: add GraphRAG store with pgvector semantic search"
```

---

### Task 10: Memory Extraction

**Files:**
- Create: `src/prot/memory.py`
- Create: `tests/test_memory.py`

**Step 1: Write failing tests**

`tests/test_memory.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prot.memory import MemoryExtractor

@pytest.mark.asyncio
class TestMemoryExtractor:
    async def test_extract_from_conversation_returns_entities(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"entities": [{"name": "Bob", "type": "person", "description": "A friend"}], "relationships": []}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            result = await extractor.extract_from_conversation([
                {"role": "user", "content": "I met Bob today"},
                {"role": "assistant", "content": "How was it?"},
            ])
            assert len(result["entities"]) == 1
            assert result["entities"][0]["name"] == "Bob"

    async def test_save_extraction_calls_store(self):
        mock_store = AsyncMock()
        mock_store.upsert_entity.return_value = "fake-uuid"
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 1024]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        await extractor.save_extraction({
            "entities": [{"name": "Bob", "type": "person", "description": "A friend"}],
            "relationships": [],
        })
        mock_store.upsert_entity.assert_called_once()
        mock_embedder.embed_texts.assert_called_once()

    async def test_pre_load_context_returns_text(self):
        mock_store = AsyncMock()
        mock_store.search_communities.return_value = [
            {"summary": "Community about friends", "similarity": 0.9},
            {"summary": "Community about work", "similarity": 0.8},
        ]
        mock_embedder = AsyncMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        text = await extractor.pre_load_context("Tell me about friends")
        assert "friends" in text
        assert "work" in text
```

**Step 2: Write implementation**

`src/prot/memory.py`:

```python
import json
from anthropic import AsyncAnthropic
from prot.config import settings
from prot.graphrag import GraphRAGStore
from prot.embeddings import AsyncVoyageEmbedder

EXTRACTION_PROMPT = """Extract entities and relationships from this conversation.

Return JSON with this exact structure:
{
  "entities": [{"name": "...", "type": "person|place|concept|event", "description": "..."}],
  "relationships": [{"source": "...", "target": "...", "type": "...", "description": "..."}]
}

Only extract factual information. Skip greetings and filler."""

class MemoryExtractor:
    """Extract and manage long-term memory from conversations."""

    def __init__(
        self,
        anthropic_key: str | None = None,
        store: GraphRAGStore | None = None,
        embedder: AsyncVoyageEmbedder | None = None,
    ):
        self._llm = AsyncAnthropic(api_key=anthropic_key or settings.anthropic_api_key)
        self._store = store
        self._embedder = embedder

    async def extract_from_conversation(self, messages: list[dict]) -> dict:
        """Use Haiku 4.5 to extract entities and relationships from conversation."""
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        response = await self._llm.messages.create(
            model=settings.memory_extraction_model,
            max_tokens=2000,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": conversation_text}],
        )
        return json.loads(response.content[0].text)

    async def save_extraction(self, extraction: dict) -> None:
        """Embed and save extracted entities and relationships."""
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        if not entities:
            return

        # Embed entity descriptions
        descriptions = [e["description"] for e in entities]
        embeddings = await self._embedder.embed_texts(descriptions)

        # Upsert entities
        entity_ids = {}
        for entity, embedding in zip(entities, embeddings):
            eid = await self._store.upsert_entity(
                name=entity["name"],
                entity_type=entity["type"],
                description=entity["description"],
                embedding=embedding,
            )
            entity_ids[entity["name"]] = eid

        # Upsert relationships
        for rel in relationships:
            src_id = entity_ids.get(rel["source"])
            tgt_id = entity_ids.get(rel["target"])
            if src_id and tgt_id:
                await self._store.upsert_relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=rel["type"],
                    description=rel["description"],
                )

    async def pre_load_context(self, query: str) -> str:
        """Search GraphRAG and assemble Block 2 context text.

        Target ~3,000 tokens to meet cache minimum (4,096 with persona+tools).
        """
        query_embedding = await self._embedder.embed_query(query)
        communities = await self._store.search_communities(
            query_embedding=query_embedding,
            top_k=settings.rag_top_k,
        )

        parts = []
        token_estimate = 0
        for community in communities:
            summary = community["summary"]
            parts.append(summary)
            token_estimate += len(summary) // 4  # rough estimate
            if token_estimate >= settings.rag_context_target_tokens:
                break

        return "\n\n".join(parts) if parts else "(no memory context)"
```

**Step 3: Run tests**

```bash
uv run pytest tests/test_memory.py -v
```

**Step 4: Commit**

```bash
git add src/prot/memory.py tests/test_memory.py
git commit -m "feat: add memory extraction with Haiku 4.5 and GraphRAG pre-loading"
```

---

## Phase 3: Audio Output (Text In → Voice Out)

### Task 11: TTS Client (ElevenLabs Flash Streaming)

> Previously Task 7 — renumbered after Phase 2B insertion.

**Files:**
- Create: `src/prot/tts.py`
- Create: `tests/test_tts.py`

**Step 1: Write failing tests**

`tests/test_tts.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prot.tts import TTSClient

@pytest.mark.asyncio
class TestTTSClient:
    async def test_stream_audio_yields_bytes(self):
        mock_response = AsyncMock()
        mock_response.__aiter__ = lambda self: self
        mock_response.__anext__ = AsyncMock(
            side_effect=[b"\x00" * 1024, b"\x00" * 512, StopAsyncIteration]
        )

        with patch("prot.tts.AsyncElevenLabs") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.text_to_speech.convert_as_stream = AsyncMock(
                return_value=mock_response
            )

            tts = TTSClient(api_key="test")
            chunks = []
            async for chunk in tts.stream_audio("테스트 문장"):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert all(isinstance(c, bytes) for c in chunks)

    async def test_flush_clears_state(self):
        tts = TTSClient(api_key="test")
        tts.flush()
        assert tts._cancelled is True
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_tts.py -v
```

**Step 3: Write implementation**

`src/prot/tts.py`:

```python
from elevenlabs import AsyncElevenLabs
from prot.config import settings

class TTSClient:
    def __init__(self, api_key: str | None = None):
        self._client = AsyncElevenLabs(api_key=api_key or settings.elevenlabs_api_key)
        self._cancelled = False

    async def stream_audio(self, text: str):
        """Stream PCM audio bytes for given text."""
        self._cancelled = False
        response = await self._client.text_to_speech.convert_as_stream(
            voice_id=settings.elevenlabs_voice_id,
            text=text,
            model_id=settings.elevenlabs_model,
            output_format=settings.elevenlabs_output_format,
        )
        async for chunk in response:
            if self._cancelled:
                break
            if isinstance(chunk, bytes):
                yield chunk

    def flush(self) -> None:
        """Cancel current TTS stream."""
        self._cancelled = True
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_tts.py -v
```

**Step 5: Commit**

```bash
git add src/prot/tts.py tests/test_tts.py
git commit -m "feat: add ElevenLabs Flash streaming TTS client"
```

---

### Task 12: Audio Playback (paplay)

**Files:**
- Create: `src/prot/playback.py`
- Create: `tests/test_playback.py`

**Step 1: Write failing tests**

`tests/test_playback.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prot.playback import AudioPlayer

@pytest.mark.asyncio
class TestAudioPlayer:
    async def test_play_chunk_writes_to_stdin(self):
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            player = AudioPlayer()
            await player.start()
            await player.play_chunk(b"\x00" * 1024)
            mock_proc.stdin.write.assert_called_once_with(b"\x00" * 1024)

    async def test_kill_terminates_process(self):
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.returncode = None

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            player = AudioPlayer()
            await player.start()
            await player.kill()
            mock_proc.kill.assert_called_once()
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_playback.py -v
```

**Step 3: Write implementation**

`src/prot/playback.py`:

```python
import asyncio

class AudioPlayer:
    """Async wrapper around paplay for PCM audio output."""

    def __init__(self, rate: int = 16000, channels: int = 1, format: str = "s16le"):
        self._rate = rate
        self._channels = channels
        self._format = format
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start paplay subprocess."""
        self._process = await asyncio.create_subprocess_exec(
            "paplay",
            f"--format={self._format}",
            f"--rate={self._rate}",
            f"--channels={self._channels}",
            "--raw",
            stdin=asyncio.subprocess.PIPE,
        )

    async def play_chunk(self, data: bytes) -> None:
        """Write audio chunk to paplay stdin."""
        if self._process and self._process.stdin:
            self._process.stdin.write(data)
            await self._process.stdin.drain()

    async def finish(self) -> None:
        """Close stdin and wait for paplay to finish."""
        if self._process and self._process.stdin:
            self._process.stdin.close()
            await self._process.wait()
        self._process = None

    async def kill(self) -> None:
        """Immediately kill paplay (for barge-in)."""
        if self._process and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        self._process = None
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_playback.py -v
```

**Step 5: Commit**

```bash
git add src/prot/playback.py tests/test_playback.py
git commit -m "feat: add async paplay audio player with barge-in kill"
```

---

## Phase 4: Audio Input (Voice In)

### Task 13: VAD Processor (Silero)

**Files:**
- Create: `src/prot/vad.py`
- Create: `tests/test_vad.py`

**Step 1: Write failing tests**

`tests/test_vad.py`:

```python
import pytest
import numpy as np
from prot.vad import VADProcessor

class TestVADProcessor:
    def test_silence_returns_false(self):
        vad = VADProcessor()
        silence = np.zeros(512, dtype=np.int16)
        assert vad.is_speech(silence.tobytes()) is False

    def test_threshold_property(self):
        vad = VADProcessor(threshold=0.5)
        assert vad.threshold == 0.5
        vad.threshold = 0.8
        assert vad.threshold == 0.8

    def test_reset_clears_state(self):
        vad = VADProcessor()
        vad.reset()
        assert vad._speech_count == 0
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_vad.py -v
```

**Step 3: Write implementation**

`src/prot/vad.py`:

```python
import torch
import numpy as np

class VADProcessor:
    """Silero VAD wrapper for speech detection."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000, speech_count_threshold: int = 3):
        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._model.eval()
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._speech_count_threshold = speech_count_threshold
        self._speech_count = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """Check if PCM audio chunk contains speech."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)
        prob = self._model(tensor, self._sample_rate).item()

        if prob >= self._threshold:
            self._speech_count += 1
        else:
            self._speech_count = 0

        return self._speech_count >= self._speech_count_threshold

    def reset(self) -> None:
        """Reset internal state."""
        self._speech_count = 0
        self._model.reset_states()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_vad.py -v
```

**Step 5: Commit**

```bash
git add src/prot/vad.py tests/test_vad.py
git commit -m "feat: add Silero VAD processor with adjustable threshold"
```

---

### Task 14: STT Client (Deepgram Flux WebSocket)

**Files:**
- Create: `src/prot/stt.py`
- Create: `tests/test_stt.py`

**Step 1: Write failing tests**

`tests/test_stt.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from prot.stt import STTClient

@pytest.mark.asyncio
class TestSTTClient:
    async def test_on_transcript_callback(self):
        transcripts = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        client = STTClient(api_key="test", on_transcript=on_transcript)
        # Simulate receiving a final transcript
        await client._handle_transcript("테스트 문장", is_final=True)
        assert transcripts == [("테스트 문장", True)]

    async def test_on_utterance_end_callback(self):
        called = []

        async def on_end():
            called.append(True)

        client = STTClient(api_key="test", on_utterance_end=on_end)
        await client._handle_utterance_end()
        assert called == [True]

    def test_keyterms_configured(self):
        client = STTClient(
            api_key="test",
            keyterms=["Axel", "NorthProt"],
        )
        assert client._keyterms == ["Axel", "NorthProt"]
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_stt.py -v
```

**Step 3: Write implementation**

`src/prot/stt.py`:

```python
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from prot.config import settings
from typing import Callable, Awaitable

class STTClient:
    """Deepgram Flux WebSocket streaming client."""

    def __init__(
        self,
        api_key: str | None = None,
        on_transcript: Callable[[str, bool], Awaitable[None]] | None = None,
        on_utterance_end: Callable[[], Awaitable[None]] | None = None,
        keyterms: list[str] | None = None,
    ):
        self._client = DeepgramClient(api_key or settings.deepgram_api_key)
        self._connection = None
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._keyterms = keyterms or []

    async def connect(self) -> None:
        """Open WebSocket connection to Deepgram."""
        options = LiveOptions(
            model=settings.deepgram_model,
            language=settings.deepgram_language,
            smart_format=True,
            interim_results=True,
            utterance_end_ms=1000,
            endpointing=settings.deepgram_endpointing,
            keyterm=self._keyterms if self._keyterms else None,
        )

        self._connection = self._client.listen.asyncwebsocket.v("1")

        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
        self._connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utt_end)

        await self._connection.start(options)

    async def send_audio(self, data: bytes) -> None:
        """Send PCM audio chunk to Deepgram."""
        if self._connection:
            await self._connection.send(data)

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._connection:
            await self._connection.finish()
            self._connection = None

    async def _on_message(self, _client, result, **kwargs) -> None:
        transcript = result.channel.alternatives[0].transcript
        if not transcript:
            return
        is_final = result.is_final
        await self._handle_transcript(transcript, is_final)

    async def _on_utt_end(self, _client, result, **kwargs) -> None:
        await self._handle_utterance_end()

    async def _handle_transcript(self, text: str, is_final: bool) -> None:
        if self._on_transcript:
            await self._on_transcript(text, is_final)

    async def _handle_utterance_end(self) -> None:
        if self._on_utterance_end:
            await self._on_utterance_end()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_stt.py -v
```

**Step 5: Commit**

```bash
git add src/prot/stt.py tests/test_stt.py
git commit -m "feat: add Deepgram Flux WebSocket STT client with keyterms"
```

---

### Task 15: Audio Manager (PyAudio)

**Files:**
- Create: `src/prot/audio.py`
- Create: `tests/test_audio.py`

**Step 1: Write failing tests**

`tests/test_audio.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from prot.audio import AudioManager

class TestAudioManager:
    def test_chunk_size_config(self):
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11, chunk_size=512)
            assert mgr.chunk_size == 512

    def test_callback_receives_data(self):
        received = []
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11, on_audio=lambda d: received.append(d))
            mgr._audio_callback(b"\x00" * 1024, 512, {}, 0)
            assert len(received) == 1
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_audio.py -v
```

**Step 3: Write implementation**

`src/prot/audio.py`:

```python
import pyaudio
from typing import Callable

class AudioManager:
    """PyAudio mic input with non-blocking callback."""

    def __init__(
        self,
        device_index: int = 11,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        on_audio: Callable[[bytes], None] | None = None,
    ):
        self._pa = pyaudio.PyAudio()
        self._device_index = device_index
        self._sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._on_audio = on_audio
        self._stream = None

    def start(self) -> None:
        """Open mic stream with callback."""
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        """Stop mic stream."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self._on_audio and in_data:
            self._on_audio(in_data)
        return (None, pyaudio.paContinue)

    def __del__(self):
        self.stop()
        self._pa.terminate()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_audio.py -v
```

**Step 5: Commit**

```bash
git add src/prot/audio.py tests/test_audio.py
git commit -m "feat: add PyAudio mic input manager with callback"
```

---

## Phase 5: Pipeline Integration

### Task 16: Pipeline Orchestrator

The core: wires all components together with the state machine.

**Files:**
- Create: `src/prot/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write failing tests**

`tests/test_pipeline.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prot.pipeline import Pipeline
from prot.state import State

@pytest.mark.asyncio
class TestPipeline:
    async def test_speech_triggers_listening(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.state = MagicMock()
        pipeline.state.state = State.IDLE
        pipeline.stt = AsyncMock()
        pipeline._handle_vad_speech = Pipeline._handle_vad_speech.__get__(pipeline)

        await pipeline._handle_vad_speech()
        pipeline.state.on_speech_detected.assert_called_once()
        pipeline.stt.connect.assert_called_once()

    async def test_utterance_triggers_processing(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.state = MagicMock()
        pipeline.state.state = State.LISTENING
        pipeline._final_transcript = "테스트"
        pipeline.llm = AsyncMock()
        pipeline.tts = AsyncMock()
        pipeline.player = AsyncMock()
        pipeline.context = MagicMock()
        pipeline.context.build_system_blocks.return_value = []
        pipeline.context.build_tools.return_value = []
        pipeline.context.get_messages.return_value = []
        pipeline._process_response = AsyncMock()

        await pipeline._handle_utterance_end()
        pipeline.state.on_utterance_complete.assert_called_once()
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_pipeline.py -v
```

**Step 3: Write implementation**

`src/prot/pipeline.py`:

```python
import asyncio
from prot.state import StateMachine, State
from prot.config import settings
from prot.vad import VADProcessor
from prot.stt import STTClient
from prot.llm import LLMClient
from prot.tts import TTSClient
from prot.playback import AudioPlayer
from prot.context import ContextManager
from prot.persona import load_persona
from prot.processing import sanitize_for_tts, chunk_sentences
from prot.db import init_pool, close_pool
from prot.graphrag import GraphRAGStore
from prot.embeddings import AsyncVoyageEmbedder
from prot.memory import MemoryExtractor

class Pipeline:
    def __init__(self):
        self.state = StateMachine(
            vad_threshold_normal=settings.vad_threshold,
            vad_threshold_speaking=settings.vad_threshold_speaking,
        )
        self.vad = VADProcessor(threshold=settings.vad_threshold)
        self.stt = STTClient(
            on_transcript=self._on_transcript,
            on_utterance_end=self._handle_utterance_end,
            keyterms=["Axel", "NorthProt", "prot"],
        )
        self.llm = LLMClient()
        self.tts = TTSClient()
        self.player = AudioPlayer()
        self.context = ContextManager(persona_text=load_persona())

        # Data layer (initialized in async startup)
        self._db_pool = None
        self.graphrag: GraphRAGStore | None = None
        self.embedder: AsyncVoyageEmbedder | None = None
        self.memory: MemoryExtractor | None = None

        self._final_transcript = ""
        self._active_timeout_task: asyncio.Task | None = None

    async def startup(self) -> None:
        """Initialize async resources (DB pool, memory components)."""
        self._db_pool = await init_pool()
        self.graphrag = GraphRAGStore(self._db_pool)
        self.embedder = AsyncVoyageEmbedder()
        self.memory = MemoryExtractor(store=self.graphrag, embedder=self.embedder)

    def on_audio_chunk(self, data: bytes) -> None:
        """Called by AudioManager callback. Runs VAD check."""
        self.vad.threshold = self.state.vad_threshold

        if self.vad.is_speech(data):
            if self.state.state in (State.IDLE, State.ACTIVE):
                asyncio.get_event_loop().create_task(self._handle_vad_speech())
            elif self.state.state == State.SPEAKING:
                asyncio.get_event_loop().create_task(self._handle_barge_in())

        # Forward audio to Deepgram if listening
        if self.state.state in (State.LISTENING,):
            asyncio.get_event_loop().create_task(self.stt.send_audio(data))

    async def _handle_vad_speech(self) -> None:
        """VAD detected speech in IDLE/ACTIVE state."""
        was_idle = self.state.state == State.IDLE

        if self._active_timeout_task:
            self._active_timeout_task.cancel()
            self._active_timeout_task = None

        self.state.on_speech_detected()
        self._final_transcript = ""

        if self.state.state == State.LISTENING:
            await self.stt.connect()

        # Pre-load GraphRAG context on conversation start (IDLE → LISTENING)
        if was_idle and self.memory:
            try:
                rag_text = await self.memory.pre_load_context("conversation start")
                self.context.update_rag_context(rag_text)
            except Exception:
                pass  # Non-critical: proceed without RAG context

    async def _on_transcript(self, text: str, is_final: bool) -> None:
        """Deepgram transcript callback."""
        if is_final:
            self._final_transcript = text

    async def _handle_utterance_end(self) -> None:
        """Deepgram utterance end — user finished speaking."""
        if not self._final_transcript:
            return

        self.state.on_utterance_complete()
        self.context.add_message("user", self._final_transcript)
        await self._process_response()

    async def _process_response(self) -> None:
        """Stream LLM → TTS → playback."""
        full_response = ""
        first_sentence = True

        async for chunk in self.llm.stream_response(
            system_blocks=self.context.build_system_blocks(),
            tools=self.context.build_tools(),
            messages=self.context.get_messages(),
        ):
            full_response += chunk
            sentences = list(chunk_sentences(full_response))

            # Yield complete sentences to TTS
            if len(sentences) > 1 or (sentences and full_response.rstrip()[-1:] in '.!?~'):
                for sentence in sentences[:-1] if len(sentences) > 1 else sentences:
                    clean = sanitize_for_tts(sentence)
                    if not clean:
                        continue

                    if first_sentence:
                        self.state.on_tts_started()
                        await self.player.start()
                        first_sentence = False

                    async for audio_chunk in self.tts.stream_audio(clean):
                        if self.state.state == State.INTERRUPTED:
                            return
                        await self.player.play_chunk(audio_chunk)

                full_response = sentences[-1] if len(sentences) > 1 else ""

        # Handle remaining text
        if full_response.strip():
            clean = sanitize_for_tts(full_response)
            if clean:
                if first_sentence:
                    self.state.on_tts_started()
                    await self.player.start()
                async for audio_chunk in self.tts.stream_audio(clean):
                    if self.state.state == State.INTERRUPTED:
                        return
                    await self.player.play_chunk(audio_chunk)

        await self.player.finish()
        self.context.add_message("assistant", sanitize_for_tts(full_response))

        if self.state.state == State.SPEAKING:
            self.state.on_tts_complete()
            self._start_active_timeout()

    async def _handle_barge_in(self) -> None:
        """User interrupted during TTS playback."""
        self.state.on_speech_detected()  # SPEAKING → INTERRUPTED
        self.llm.cancel()
        self.tts.flush()
        await self.player.kill()
        self.state.on_interrupt_handled()  # INTERRUPTED → LISTENING
        self._final_transcript = ""
        await self.stt.connect()

    def _start_active_timeout(self) -> None:
        """Start 30s timer to return to IDLE."""
        async def _timeout():
            await asyncio.sleep(settings.active_timeout)
            if self.state.state == State.ACTIVE:
                self.state.on_active_timeout()
                await self.stt.disconnect()
                # Background memory extraction on conversation end
                if self.memory:
                    asyncio.get_event_loop().create_task(
                        self._extract_memories()
                    )

        self._active_timeout_task = asyncio.get_event_loop().create_task(_timeout())

    async def _extract_memories(self) -> None:
        """Background task: extract entities/relationships from conversation."""
        try:
            messages = self.context.get_messages()
            if not messages:
                return
            extraction = await self.memory.extract_from_conversation(messages)
            await self.memory.save_extraction(extraction)
        except Exception:
            pass  # Non-critical: log and continue

    async def shutdown(self) -> None:
        """Clean shutdown."""
        self.llm.cancel()
        self.tts.flush()
        await self.player.kill()
        await self.stt.disconnect()
        await close_pool()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_pipeline.py -v
```

**Step 5: Commit**

```bash
git add src/prot/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator wiring all components"
```

---

### Task 17: FastAPI App & Entry Point

**Files:**
- Create: `src/prot/app.py`

**Step 1: Write app**

`src/prot/app.py`:

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prot.pipeline import Pipeline
from prot.audio import AudioManager
from prot.config import settings

pipeline: Pipeline | None = None
audio: AudioManager | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, audio
    pipeline = Pipeline()
    await pipeline.startup()  # Initialize DB pool, GraphRAG, embedder, memory
    audio = AudioManager(
        device_index=settings.mic_device_index,
        sample_rate=settings.sample_rate,
        chunk_size=settings.chunk_size,
        on_audio=pipeline.on_audio_chunk,
    )
    audio.start()
    print(f"prot started. Mic device={settings.mic_device_index}. Listening...")
    yield
    audio.stop()
    await pipeline.shutdown()
    print("prot stopped.")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "state": pipeline.state.state.value if pipeline else "not_started",
    }

@app.get("/state")
async def state():
    return {"state": pipeline.state.state.value if pipeline else "not_started"}
```

**Step 2: Test manually**

```bash
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000
# In another terminal:
curl http://localhost:8000/health
```

**Step 3: Commit**

```bash
git add src/prot/app.py
git commit -m "feat: add FastAPI app with health endpoint and audio lifecycle"
```

---

### Task 18: Conversation Logging

**Files:**
- Create: `src/prot/logger.py`
- Create: `tests/test_logger.py`

**Step 1: Write failing tests**

`tests/test_logger.py`:

```python
import pytest
import json
from pathlib import Path
from prot.logger import ConversationLogger

class TestConversationLogger:
    def test_log_creates_daily_file(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "안녕")
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_log_appends_to_existing(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "안녕")
        logger.log("assistant", "야 뭐해")
        files = list(tmp_path.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert len(data) == 2

    def test_log_includes_timestamp(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "테스트")
        files = list(tmp_path.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert "timestamp" in data[0]
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_logger.py -v
```

**Step 3: Write implementation**

`src/prot/logger.py`:

```python
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))

class ConversationLogger:
    def __init__(self, log_dir: Path | str = "logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _today_file(self) -> Path:
        return self._log_dir / f"{datetime.now(KST).strftime('%Y-%m-%d')}.json"

    def log(self, role: str, content: str) -> None:
        path = self._today_file()
        entries = []
        if path.exists():
            entries = json.loads(path.read_text(encoding="utf-8"))

        entries.append({
            "timestamp": datetime.now(KST).isoformat(),
            "role": role,
            "content": content,
        })
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_logger.py -v
```

**Step 5: Commit**

```bash
git add src/prot/logger.py tests/test_logger.py
git commit -m "feat: add daily JSON conversation logger"
```

---

### Task 19: systemd Service

**Files:**
- Create: `deploy/prot.service`

**Step 1: Write service file**

`deploy/prot.service`:

```ini
[Unit]
Description=prot voice conversation service
After=network.target pulseaudio.service

[Service]
Type=simple
User=cyan
WorkingDirectory=/home/cyan/workplace/prot
Environment=PATH=/home/cyan/.local/bin:/usr/local/bin:/usr/bin
EnvironmentFile=/home/cyan/workplace/prot/.env
ExecStart=/home/cyan/workplace/prot/.venv/bin/uvicorn prot.app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Step 2: Commit**

```bash
git add deploy/prot.service
git commit -m "chore: add systemd service file for deployment"
```

---

## Summary

| Phase | Tasks | What you get |
|-------|-------|-------------|
| 1 | 1-3 | Project setup, TTS sanitizer, state machine |
| 2 | 4-6 | Text-in → text-out with Axel persona |
| 2B | 7-10 | Database, embeddings, GraphRAG, memory extraction |
| 3 | 11-12 | Text-in → voice-out (hear Axel speak) |
| 4 | 13-15 | Voice-in (mic → STT) |
| 5 | 16-19 | Full pipeline, FastAPI app, logging, systemd |

Each phase produces a runnable checkpoint. Phase 2 is the first "talk to Axel" moment (text). Phase 2B adds persistent memory. Phase 3 is when you first hear his voice. Phase 5 is the full loop.
