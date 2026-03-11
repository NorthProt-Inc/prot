# prot Operations Guide

> Installation, configuration, running, debugging, operations, and troubleshooting guide for the Axel persona real-time voice conversation system.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Running](#running)
4. [Debugging](#debugging)
5. [Operations](#operations)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

| Item | Version | Notes |
|------|---------|-------|
| Python | 3.12+ | Verify with `python3 --version` |
| [uv](https://docs.astral.sh/uv/) | latest | Package manager |
| PulseAudio | — | For `paplay` audio output. PipeWire-PulseAudio compatible |
| PostgreSQL | 15+ | Required for memory/RAG features (optional) |
| pgvector | 0.5+ | PostgreSQL vector extension (optional) |

### Dependencies

```bash
# Production dependencies
uv sync

# With dev dependencies (pytest, pytest-asyncio, pytest-cov)
uv sync --extra dev
```

---

## Configuration

### Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file to set API keys and configuration values. See the tables below for details.

#### API Keys (Required)

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key — used for Claude LLM calls |
| `ELEVENLABS_API_KEY` | ElevenLabs API key — used for both STT (Scribe v2) and TTS |

#### Audio / VAD

| Variable | Default | Description |
|----------|---------|-------------|
| `MIC_DEVICE_INDEX` | *(system default)* | PyAudio input device index |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `CHUNK_SIZE` | `512` | Audio chunk size (bytes) |
| `BARGE_IN_ENABLED` | `false` | Enable barge-in (interrupting during SPEAKING) |
| `VAD_THRESHOLD` | `0.5` | VAD speech detection threshold (IDLE / ACTIVE states) |
| `VAD_THRESHOLD_SPEAKING` | `0.8` | VAD threshold (SPEAKING state, barge-in detection) |
| `VAD_PREBUFFER_CHUNKS` | `8` | VAD pre-buffer chunk count |

#### STT / LLM / TTS

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_LANGUAGE` | `ko` | STT recognition language |
| `STT_SILENCE_THRESHOLD_SECS` | `3.0` | STT silence detection timeout (seconds) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model ID |
| `CLAUDE_MAX_TOKENS` | `4096` | Claude max output tokens |
| `CLAUDE_EFFORT` | `high` | Claude thinking effort (`low` / `medium` / `high`) |
| `COMPACTION_TRIGGER` | `50000` | Token threshold to trigger server-side compaction |
| `TOOL_CLEAR_TRIGGER` | `30000` | Token threshold for tool result clearing |
| `TOOL_CLEAR_KEEP` | `3` | Number of recent tool results to keep after clearing |
| `THINKING_KEEP_TURNS` | `2` | Number of recent thinking turns to keep after clearing |
| `ELEVENLABS_VOICE_ID` | `s3lKyrFAzTUpzy3ZLwbM` | ElevenLabs voice ID |
| `ELEVENLABS_MODEL` | `eleven_v3` | ElevenLabs TTS model |
| `ELEVENLABS_OUTPUT_FORMAT` | `pcm_24000` | TTS output audio format |
| `TTS_SENTENCE_SILENCE_MS` | `200` | Silence gap between sentences (ms) |

#### Home Assistant

| Variable | Default | Description |
|----------|---------|-------------|
| `HASS_URL` | `http://localhost:8123` | Home Assistant URL |
| `HASS_TOKEN` | — | Home Assistant long-lived access token |
| `HASS_AGENT_ID` | `conversation.google_ai_conversation` | HA conversation agent entity ID |

#### Database / Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://prot:prot@localhost:5432/prot` | PostgreSQL connection string |
| `DB_POOL_MIN` | `2` | DB connection pool min size |
| `DB_POOL_MAX` | `10` | DB connection pool max size |
| `VOYAGE_API_KEY` | — | Voyage AI embedding API key |
| `VOYAGE_MODEL` | `voyage-4-large` | Embedding model |
| `RERANK_MODEL` | `rerank-2.5` | Voyage reranker model |
| `RERANK_TOP_K` | `5` | Reranker top-K results |
| `MEMORY_EXTRACTION_MODEL` | `claude-haiku-4-5-20251001` | Model for memory extraction |
| `RAG_CONTEXT_TARGET_TOKENS` | `4096` | RAG context target token budget |
| `RAG_TOP_K` | `10` | RAG retrieval top-K candidates |
| `PAUSE_AFTER_COMPACTION` | `true` | Pause pipeline after server compaction |
| `DECAY_BASE_RATE` | `0.002` | Memory time-decay base rate |
| `DECAY_MIN_RETENTION` | `0.1` | Minimum memory retention score |

#### CORS / Logging / Timers

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins (JSON list) |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |
| `ACTIVE_TIMEOUT` | `30` | Seconds in ACTIVE state before transitioning to IDLE |

### Database Setup (Optional)

PostgreSQL + pgvector is required for memory / GraphRAG features. The app runs without a database — basic conversation works fine.

```bash
# 1. Create the database
createdb prot

# 2. Install extensions
psql prot -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql prot -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

The schema is auto-applied on application startup. For manual application:

```bash
psql prot < src/prot/schema.sql
```

#### Memory Table Structure

| Table | Purpose |
|-------|---------|
| `semantic_memories` | Facts, knowledge, preferences (subject-predicate-object triples) |
| `episodic_memories` | Conversation episode summaries, topics, emotional tone |
| `emotional_memories` | Emotional context, bonds |
| `procedural_memories` | Habits, behavioral patterns |
| `conversation_messages` | Persistent conversation message storage |

All memory tables have a `vector(1024)` column with HNSW indexes.

### Audio Device Setup

```bash
python3 -c "
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'{i}: {info[\"name\"]}')
pa.terminate()
"
```

Set the desired device number in `MIC_DEVICE_INDEX`.

### Output Volume Control

When using headphones, lower the volume for hearing protection.

```bash
# Check current volume
pactl get-sink-volume @DEFAULT_SINK@

# Set volume (recommended for headphones: 20-30%)
pactl set-sink-volume @DEFAULT_SINK@ 25%

# Check output devices and active ports
pactl list sinks | grep -E "Name:|Description:|Active Port:|State:"
```

---

## Running

### Development Mode

```bash
# Dev launcher — auto-kills stale port processes before starting
./scripts/run.sh

# Or manual start (hot reload)
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --reload
```

`scripts/run.sh` automatically terminates any process on the specified port (default 8000) before starting uvicorn. Change the port via the `PORT` environment variable.

### Production Mode

```bash
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --log-level info
```

### systemd (user service)

```bash
# 1. Copy the service file
cp deploy/prot.service ~/.config/systemd/user/

# 2. Register and start
systemctl --user daemon-reload
systemctl --user enable --now prot

# 3. Check status
systemctl --user status prot

# 4. Follow logs
journalctl --user -u prot -f

# 5. Restart
systemctl --user restart prot
```

> **Note**: `prot.service` depends on `pipewire-pulse.service`. PulseAudio or PipeWire-PulseAudio must be running.

---

## Debugging

### Health Check

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
# {
#     "status": "ok",
#     "state": "idle"
# }
```

### State Check

```bash
curl -s http://localhost:8000/state | python3 -m json.tool
# {
#     "state": "idle"
# }
```

### Runtime Diagnostics

```bash
curl -s http://localhost:8000/diagnostics | python3 -m json.tool
# {
#     "state": "idle",
#     "background_tasks": 0,
#     "active_timeout": false,
#     "db_pool_size": 10,
#     "db_pool_free": 10,
#     ...
# }
```

### Memory Profiling

```bash
PROT_TRACEMALLOC=1 uv run uvicorn prot.app:app --port 8000
curl -s http://localhost:8000/memory | python3 -m json.tool
```

### Change Log Level

Set `LOG_LEVEL=DEBUG` in `.env` and restart the service:

```bash
systemctl --user restart prot
```

### Testing

```bash
# Unit tests (no API keys needed)
uv run pytest

# Integration tests (requires real API keys)
uv run pytest -m integration

# Coverage report
uv run pytest --cov=prot --cov-report=term-missing
```

---

## Operations

### State Machine Flow

```
                    speech_detected        utterance_complete       tts_started
          IDLE ──────────────────► LISTENING ───────────────► PROCESSING ──────────► SPEAKING
           ▲                          ▲                           ▲                   │  │  │
           │                          │                           │                   │  │  │
           │  active_timeout          │  interrupt_handled        │  tool_iteration   │  │  │
           │                          │                           │                   │  │  │
         ACTIVE ◄─────────────────────┼───────────────── INTERRUPTED ◄────── barge_in┘  │  │
           ▲                          │                                                  │  │
           └──────────────────────────┼──────────────────────────────── tts_complete ────┘  │
                                      └──── speech_detected (from ACTIVE) ─────────────────┘
```

| State | Description |
|-------|-------------|
| **IDLE** | Waiting. Transitions to LISTENING when VAD detects speech |
| **LISTENING** | STT WebSocket connected, converting speech to text |
| **PROCESSING** | Claude generating response (includes tool loop) |
| **SPEAKING** | TTS audio playing. Can be interrupted via barge-in |
| **ACTIVE** | Response complete, waiting for follow-up speech (30s timeout) |
| **INTERRUPTED** | User interrupted during SPEAKING. STT reconnects, transitions to LISTENING |

### Dual-Channel Architecture

The system supports two interaction channels:

- **Voice** (`channel=voice`): Full audio pipeline with FSM, VAD, STT, TTS
- **Chat** (`channel=chat`): WebSocket `/chat` endpoint, text-only, per-connection engine

Both channels share the same ConversationEngine core, LLM, tools, and memory system. The `channel` parameter is passed to the persona via Block 3 of the system prompt, allowing channel-aware behavior.

### Monitoring Checklist

| Item | How to Check | Abnormal Condition |
|------|-------------|-------------------|
| Service alive | `GET /health` | `status` ≠ `ok` |
| Background tasks | `GET /diagnostics` → `background_tasks` | Steadily increasing → possible memory leak |
| DB connection pool | `GET /diagnostics` → `db_pool_free` | `0` → connection pool exhausted |
| STT/TTS errors | systemd logs | Frequent `STT connect failed` / `TTS stream failed` |
| State stuck | `GET /state` | Stuck in PROCESSING / SPEAKING for extended time |

### DB Backup

```bash
pg_dump prot > prot_backup_$(date +%Y%m%d).sql
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Service won't start | API keys not set | Check `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY` in `.env` |
| No microphone input | Wrong device index | Verify `MIC_DEVICE_INDEX`, check PulseAudio is running |
| Speech not recognized | ElevenLabs connection failure | Check API key validity, network, adjust `VAD_THRESHOLD` |
| No audio output | PulseAudio not installed | Verify `paplay --version`, check output device |
| DB connection failed | PostgreSQL not running | Check `DATABASE_URL`, run `pg_isready`, verify pgvector extension |
| Memory features disabled | Intended behavior | Normal when `VOYAGE_API_KEY` is not set — basic conversation works without DB |
| Barge-in unstable | Threshold too low | Increase `VAD_THRESHOLD_SPEAKING` (higher = harder to trigger barge-in) |
| Slow responses | LLM thinking overhead | Check `CLAUDE_EFFORT` (`low` / `medium` / `high`) |
| `SEGV` / crash | PyAudio device conflict | Check `journalctl --user -u prot` logs, try a different device index |
| Port conflict | Stale process on port | Use `./scripts/run.sh` (auto-cleanup) or `lsof -ti :8000 \| xargs kill` |
| FSM stuck | Abnormal state transition | Restart service, check `force_recovery()` in logs |
| Chat WebSocket drops | Connection timeout | Check `BusyError` in logs — concurrent messages not supported per connection |
