<div align="center">

# prot

**Real-time voice conversation system with the Axel persona**

A streaming audio pipeline that captures speech, understands context through a 4-layer memory system,
and responds with natural voice — all with sub-second latency. Also supports text chat via WebSocket.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://docs.astral.sh/uv/)

</div>

---

## Features

- **Streaming voice pipeline** — Mic → Silero VAD → ElevenLabs STT → Claude → ElevenLabs TTS → Speaker, with producer-consumer audio queue for gapless playback
- **Text chat via WebSocket** — `/chat` endpoint with per-connection ConversationEngine, shared memory and tool access
- **Dual-channel architecture** — Shared ConversationEngine core for both voice and text, with channel-aware persona behavior
- **6-state FSM** — IDLE / LISTENING / PROCESSING / SPEAKING / ACTIVE / INTERRUPTED with barge-in support for natural turn-taking
- **4-layer long-term memory** — Semantic, episodic, emotional, and procedural memory stored in PostgreSQL + pgvector with HNSW indexes and time-decay scoring
- **Compaction-driven memory extraction** — No per-exchange overhead; memories are extracted on server-side compaction events and shutdown via Haiku/Flash
- **RAG with reranking** — Voyage AI embeddings (voyage-4-large) + rerank-2.5 for high-relevance memory recall
- **Prompt caching** — 3-block system prompt layout (persona → RAG context → dynamic) optimized for Anthropic cache hits
- **Server-side compaction** — Anthropic context management API handles thinking clearing, tool result clearing, and compaction automatically
- **Agentic tool loop** — Up to 3 tool-use rounds per response with Home Assistant delegation and web search
- **Session timeline** — Per-request temporal context with session start time and recent turn timestamps
- **Structured logging** — Modular subsystem with structured formatters, handlers, and function tracing

---

## Architecture

```mermaid
graph TD
    subgraph Input
        MIC["Microphone<br/><sub>PyAudio</sub>"]
        VAD["Silero VAD<br/><sub>Speech Detection</sub>"]
    end

    subgraph Core
        SM["State Machine<br/><sub>6-state FSM</sub>"]
        ENGINE["ConversationEngine<br/><sub>Shared Core</sub>"]
        STT["ElevenLabs Scribe v2<br/><sub>WebSocket Realtime</sub>"]
        LLM["Claude Sonnet 4.6<br/><sub>Adaptive Thinking</sub>"]
    end

    subgraph Channels
        VOICE["Voice Pipeline<br/><sub>Concurrent TTS</sub>"]
        CHAT["Text Chat<br/><sub>WebSocket /chat</sub>"]
    end

    subgraph Output
        TTS["ElevenLabs TTS<br/><sub>Streaming</sub>"]
        PLAY["paplay<br/><sub>PulseAudio</sub>"]
    end

    subgraph Tools
        HASS["Home Assistant<br/><sub>Conversation API</sub>"]
        WEB["Web Search"]
    end

    subgraph Memory
        MEM["Memory Extractor<br/><sub>Haiku / Flash</sub>"]
        EMB["Voyage AI<br/><sub>voyage-4-large</sub>"]
        DB[("PostgreSQL<br/><sub>pgvector</sub>")]
        RERANK["Reranker<br/><sub>rerank-2.5</sub>"]
        CTX["Context Builder<br/><sub>3-block layout</sub>"]
    end

    MIC --> VAD --> SM --> STT --> ENGINE
    CHAT --> ENGINE
    ENGINE --> LLM
    ENGINE -->|voice| VOICE --> TTS --> PLAY
    ENGINE -->|text| CHAT
    LLM -->|tool_use| HASS & WEB -->|tool_result| LLM
    LLM -->|post-compaction| MEM --> EMB --> DB
    DB --> RERANK --> CTX -->|system prompt| LLM
```

### State Machine (Voice Pipeline)

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> LISTENING : speech_detected
    LISTENING --> PROCESSING : utterance_complete
    PROCESSING --> SPEAKING : tts_started
    SPEAKING --> ACTIVE : tts_complete
    SPEAKING --> INTERRUPTED : barge_in
    SPEAKING --> PROCESSING : tool_iteration
    ACTIVE --> LISTENING : speech_detected
    ACTIVE --> IDLE : active_timeout
    INTERRUPTED --> LISTENING : interrupt_handled
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Runtime** | Python 3.12+, FastAPI, uvicorn | Async HTTP server + lifespan |
| **Audio** | PyAudio, PulseAudio (paplay) | Mic capture + speaker output |
| **VAD** | Silero VAD | Speech / silence detection |
| **STT** | ElevenLabs Scribe v2 | WebSocket realtime transcription |
| **LLM** | Claude Sonnet 4.6 (Anthropic) | Conversation + tool use + thinking |
| **TTS** | ElevenLabs v3 | Streaming text-to-speech |
| **Embeddings** | Voyage AI voyage-4-large | 1024-dim memory embeddings |
| **Reranker** | Voyage AI rerank-2.5 | RAG result re-scoring |
| **Database** | PostgreSQL 15+ + pgvector | Memory storage with HNSW indexes |
| **Smart Home** | Home Assistant | Device control via conversation API |
| **Package Mgr** | uv | Fast dependency resolution |

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- PulseAudio (for `paplay` audio output)
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension *(optional — for memory features)*

### Install

```bash
git clone https://github.com/NorthProt-Inc/prot.git && cd prot
uv sync
```

### Configure

```bash
cp .env.example .env
# Fill in required API keys: ANTHROPIC_API_KEY, ELEVENLABS_API_KEY
```

### Run

```bash
# Dev launcher (auto-kills stale port processes)
./scripts/run.sh

# Or manually with hot reload
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --reload
```

### Deploy (systemd)

```bash
cp deploy/prot.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now prot
```

---

## Dev Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies |
| `uv sync --extra dev` | Install with dev dependencies (pytest, coverage) |
| `uv run pytest` | Run unit tests (no API keys needed) |
| `uv run pytest -m integration` | Run integration tests (requires API keys) |
| `uv run pytest --cov=prot --cov-report=term-missing` | Run tests with coverage report |
| `uv run uvicorn prot.app:app --reload` | Dev server with hot reload |
| `./scripts/run.sh` | Dev launcher (kills stale port, starts uvicorn) |

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for STT + TTS |

### Audio / VAD

| Variable | Default | Description |
|----------|---------|-------------|
| `MIC_DEVICE_INDEX` | *(system)* | PyAudio input device index |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `CHUNK_SIZE` | `512` | Audio chunk size (bytes) |
| `BARGE_IN_ENABLED` | `false` | Enable barge-in (interrupt during speech) |
| `VAD_THRESHOLD` | `0.5` | VAD speech detection threshold |
| `VAD_THRESHOLD_SPEAKING` | `0.8` | VAD threshold during SPEAKING state |
| `VAD_PREBUFFER_CHUNKS` | `8` | VAD pre-buffer chunk count |

### STT / LLM / TTS

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_LANGUAGE` | `ko` | STT recognition language |
| `STT_SILENCE_THRESHOLD_SECS` | `3.0` | STT silence timeout (seconds) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model ID |
| `CLAUDE_MAX_TOKENS` | `4096` | Claude max output tokens |
| `CLAUDE_EFFORT` | `high` | Thinking effort (low / medium / high) |
| `COMPACTION_TRIGGER` | `50000` | Token threshold for server-side compaction |
| `TOOL_CLEAR_TRIGGER` | `30000` | Token threshold for tool result clearing |
| `TOOL_CLEAR_KEEP` | `3` | Number of recent tool results to keep |
| `THINKING_KEEP_TURNS` | `2` | Number of recent thinking turns to keep |
| `ELEVENLABS_VOICE_ID` | `s3lKyrFAzTUpzy3ZLwbM` | ElevenLabs voice ID |
| `ELEVENLABS_MODEL` | `eleven_v3` | ElevenLabs TTS model |
| `ELEVENLABS_OUTPUT_FORMAT` | `pcm_24000` | TTS output audio format |
| `TTS_SENTENCE_SILENCE_MS` | `200` | Silence between sentences (ms) |

### Home Assistant

| Variable | Default | Description |
|----------|---------|-------------|
| `HASS_URL` | `http://localhost:8123` | Home Assistant URL |
| `HASS_TOKEN` | — | HA long-lived access token |
| `HASS_AGENT_ID` | `conversation.google_ai_conversation` | HA conversation agent entity ID |

### Database / Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://prot:prot@localhost:5432/prot` | PostgreSQL connection string |
| `DB_POOL_MIN` | `2` | Connection pool min size |
| `DB_POOL_MAX` | `10` | Connection pool max size |
| `VOYAGE_API_KEY` | — | Voyage AI API key for embeddings |
| `VOYAGE_MODEL` | `voyage-4-large` | Embedding model |
| `RERANK_MODEL` | `rerank-2.5` | Reranker model |
| `RERANK_TOP_K` | `5` | Reranker top-K results |
| `MEMORY_EXTRACTION_MODEL` | `claude-haiku-4-5-20251001` | Model for memory extraction |
| `RAG_CONTEXT_TARGET_TOKENS` | `4096` | RAG context target token budget |
| `RAG_TOP_K` | `10` | RAG retrieval top-K candidates |
| `PAUSE_AFTER_COMPACTION` | `true` | Pause pipeline after compaction |
| `DECAY_BASE_RATE` | `0.002` | Memory time-decay base rate |
| `DECAY_MIN_RETENTION` | `0.1` | Minimum memory retention score |

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ACTIVE_TIMEOUT` | `30` | Seconds before ACTIVE → IDLE |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok", "state": "..."}` |
| `GET` | `/state` | Current FSM state |
| `GET` | `/diagnostics` | Runtime diagnostics (tasks, DB pool, etc.) |
| `GET` | `/memory` | Memory allocation stats (requires `PROT_TRACEMALLOC=1`) |
| `WS` | `/chat` | Text chat — per-connection ConversationEngine with streaming responses |

### WebSocket `/chat` Protocol

**Client → Server:**
```json
{"type": "message", "content": "Hello"}
```

**Server → Client:**
```json
{"type": "chunk", "content": "..."}
{"type": "done", "full_text": "complete response"}
{"type": "error", "message": "error description"}
```

---

## Source Layout

```
src/prot/
  app.py           # FastAPI app, lifespan, HTTP + WebSocket endpoints
  engine.py        # ConversationEngine — shared core for voice + text
  pipeline.py      # Main voice pipeline orchestrator
  state.py         # 6-state FSM with barge-in support
  config.py        # Pydantic Settings (all env vars)
  context.py       # 3-block system prompt builder + session timeline
  persona.py       # Axel persona loader (data/axel.xml)
  audio.py         # PyAudio microphone capture
  vad.py           # Silero VAD speech detection
  stt.py           # ElevenLabs Scribe v2 (WebSocket STT)
  llm.py           # Claude API — streaming + tool-use loop
  hass.py          # Home Assistant conversation API
  tts.py           # ElevenLabs TTS streaming
  playback.py      # paplay audio output (producer-consumer queue)
  processing.py    # LLM → TTS → playback orchestration
  memory.py        # Compaction-driven 4-layer memory extraction
  graphrag.py      # pgvector-backed memory storage
  decay.py         # AdaptiveDecayCalculator for time-decay scoring
  embeddings.py    # Voyage AI embeddings
  reranker.py      # Voyage AI reranker
  db.py            # asyncpg connection pool + schema init
  schema.sql       # PostgreSQL schema (auto-applied on startup)
  logging/         # Structured logging subsystem

tests/             # Unit & integration tests
deploy/            # systemd service file (prot.service)
scripts/           # Dev launcher (run.sh)
data/              # Persona config (axel.xml) + runtime data
```

---

## Testing

```bash
# Unit tests (no API keys needed)
uv run pytest

# Integration tests (requires real API keys in .env)
uv run pytest -m integration

# Coverage report
uv run pytest --cov=prot --cov-report=term-missing
```

Test configuration: pytest-asyncio with `asyncio_mode = "auto"`. Test files mirror source structure: `test_<module>.py` for each module.

---

## License

[Apache-2.0](LICENSE) &copy; 2026 NorthProt Inc.
