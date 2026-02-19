# prot

Real-time voice conversation system with the Axel persona — a streaming audio pipeline that connects microphone input through VAD, STT, LLM, and TTS back to speaker output with sub-second latency.

---

## Features

- **Streaming voice pipeline** — VAD → STT → LLM → TTS with producer-consumer audio queue
- **State machine conversation flow** — 6-state FSM (IDLE, LISTENING, PROCESSING, SPEAKING, ACTIVE, INTERRUPTED) with barge-in support
- **GraphRAG long-term memory** — Entity/relationship extraction with contextual embeddings, stored in PostgreSQL + pgvector with HNSW indexes
- **Community detection** — Louvain clustering on entity graph with auto-summarization, triggered every N extractions
- **Prompt caching** — 3-block system prompt layout (persona, RAG context, dynamic) optimized for Anthropic cache hits
- **Home Assistant integration** — Tool-use loop for smart home control (get_state, call_service)
- **Web search** — Built-in web search tool via Claude API
- **Agentic tool loop** — Multi-iteration tool execution with up to 3 rounds per response
- **Structured logging** — Modular logging subsystem with structured formatters, handlers, and turn-scoped timers

---

## Architecture

```mermaid
graph TD
    MIC[Microphone<br/>PyAudio] --> VAD[Silero VAD]
    VAD --> SM[State Machine]
    SM --> STT[ElevenLabs Scribe v2<br/>WebSocket Realtime]
    STT --> LLM[Claude Sonnet 4.6<br/>Adaptive Thinking]
    LLM --> TTS[ElevenLabs TTS<br/>Streaming]
    TTS --> PLAY[paplay<br/>PulseAudio]

    LLM -->|tool_use| TOOLS{Tool Router}
    TOOLS --> HASS[Home Assistant API]
    TOOLS --> WEB[Web Search]
    TOOLS -->|tool_result| LLM

    LLM -->|background| MEM[Memory Extractor]
    MEM --> EMB[Voyage AI<br/>voyage-context-3]
    EMB --> DB[(PostgreSQL<br/>pgvector)]
    MEM -->|every 5 extractions| COMM[Community Detector<br/>Louvain]
    COMM --> EMB
    DB --> RERANK[Reranker<br/>rerank-2.5] -->|RAG context| CTX[Context Manager]
    CTX -->|system blocks| LLM
```

### State Machine

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> LISTENING : speech_detected
    LISTENING --> PROCESSING : utterance_complete
    PROCESSING --> SPEAKING : tts_started
    SPEAKING --> ACTIVE : tts_complete
    SPEAKING --> INTERRUPTED : barge-in
    SPEAKING --> PROCESSING : tool_iteration
    ACTIVE --> LISTENING : speech_detected
    ACTIVE --> IDLE : active_timeout (30s)
    INTERRUPTED --> LISTENING : interrupt_handled
```

### Source Layout

```
src/prot/              # Core application
  logging/             # Structured logging subsystem
tests/                 # Unit & integration tests
data/                  # Persona config (axel.json) + runtime data
deploy/                # systemd service files
scripts/               # Dev launcher (run.sh)
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- PulseAudio (for `paplay` audio output)
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension (optional, for memory)

### Install

```bash
git clone https://github.com/NorthProt-Inc/prot.git && cd prot
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run (development)

```bash
# Dev launcher — auto-kills stale port processes
./scripts/run.sh

# Or manually
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run (production / systemd)

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
| `uv sync --extra dev` | Install with dev dependencies |
| `uv run pytest` | Run unit tests |
| `uv run pytest -m integration` | Run integration tests (requires API keys) |
| `uv run pytest --cov=prot --cov-report=term-missing` | Run tests with coverage |
| `uv run uvicorn prot.app:app --reload` | Dev server with hot reload |
| `./scripts/run.sh` | Dev launcher (kills stale port, starts uvicorn) |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for Claude |
| `ELEVENLABS_API_KEY` | Yes | — | ElevenLabs API key for STT + TTS |
| `ELEVENLABS_VOICE_ID` | No | `s3lKyrFAzTUpzy3ZLwbM` | ElevenLabs voice ID |
| `ELEVENLABS_MODEL` | No | `eleven_v3` | ElevenLabs TTS model |
| `ELEVENLABS_OUTPUT_FORMAT` | No | `pcm_24000` | TTS output audio format |
| `VOYAGE_API_KEY` | No | — | Voyage AI API key for embeddings |
| `VOYAGE_CONTEXT_MODEL` | No | `voyage-context-3` | Contextual embedding model |
| `RERANK_MODEL` | No | `rerank-2.5` | Voyage reranker model |
| `RERANK_TOP_K` | No | `5` | Reranker top-K results |
| `MEMORY_EXTRACTION_MODEL` | No | `claude-sonnet-4-6` | Model for memory extraction |
| `MEMORY_EXTRACTION_WINDOW_TURNS` | No | `3` | Turns to extract per cycle |
| `RAG_CONTEXT_TARGET_TOKENS` | No | `4096` | RAG context target token budget |
| `RAG_TOP_K` | No | `10` | RAG retrieval top-K candidates |
| `DATABASE_URL` | No | `postgresql://prot:prot@localhost:5432/prot` | PostgreSQL connection string |
| `DB_POOL_MIN` | No | `2` | DB connection pool min size |
| `DB_POOL_MAX` | No | `10` | DB connection pool max size |
| `DB_EXPORT_DIR` | No | `data/db` | CSV export directory on shutdown |
| `HASS_URL` | No | `http://localhost:8123` | Home Assistant URL |
| `HASS_TOKEN` | No | — | Home Assistant long-lived access token |
| `MIC_DEVICE_INDEX` | No | (system default) | PyAudio input device index |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-6` | Claude model ID |
| `CLAUDE_MAX_TOKENS` | No | `4096` | Claude max output tokens |
| `CLAUDE_EFFORT` | No | `high` | Claude thinking effort (low/medium/high) |
| `CONTEXT_MAX_TURNS` | No | `10` | Sliding window size (recent turns sent to LLM) |
| `VAD_THRESHOLD` | No | `0.5` | VAD speech detection threshold |
| `VAD_THRESHOLD_SPEAKING` | No | `0.8` | VAD threshold during SPEAKING (barge-in) |
| `VAD_PREBUFFER_CHUNKS` | No | `8` | VAD pre-buffer chunk count |
| `SAMPLE_RATE` | No | `16000` | Audio sample rate (Hz) |
| `CHUNK_SIZE` | No | `512` | Audio chunk size (bytes) |
| `STT_LANGUAGE` | No | `ko` | STT recognition language |
| `ACTIVE_TIMEOUT` | No | `30` | Seconds before ACTIVE → IDLE timeout |
| `COMMUNITY_REBUILD_INTERVAL` | No | `5` | Extractions between community rebuilds |
| `COMMUNITY_MIN_ENTITIES` | No | `5` | Minimum entities before community detection runs |
| `LOG_LEVEL` | No | `INFO` | Logging level |

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check with current state |
| `GET /state` | Current state machine state |
| `GET /diagnostics` | Runtime diagnostics (tasks, DB pool, etc.) |
| `GET /memory` | Memory allocation stats (requires `PROT_TRACEMALLOC=1`) |

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

---

## License

Apache-2.0 — see [LICENSE](LICENSE)
