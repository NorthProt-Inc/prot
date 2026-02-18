# prot Architecture

## Overview
prot is a real-time voice conversation system built with Python/FastAPI.
It implements a streaming audio pipeline for low-latency voice interactions.

## Pipeline Flow
```
MIC (PyAudio) -> VAD (Silero) -> STT (ElevenLabs Scribe v2, WebSocket)
  -> LLM (Claude Opus 4.6) -> TTS (ElevenLabs streaming) -> paplay
```

## State Machine
6 states with transitions:
- **IDLE** — waiting for voice activity
- **LISTENING** — VAD detected speech, streaming to STT
- **PROCESSING** — STT complete, LLM generating response
- **SPEAKING** — TTS streaming audio playback
- **ACTIVE** — conversation session active
- **INTERRUPTED** — user interrupted during SPEAKING

Key transitions:
- IDLE -> LISTENING (VAD trigger)
- LISTENING -> PROCESSING (silence detected)
- PROCESSING -> SPEAKING (LLM response ready)
- SPEAKING -> INTERRUPTED (user speaks during playback)
- SPEAKING -> PROCESSING (tool loop iteration)

## Key Classes
- `Pipeline` (pipeline.py) — main orchestrator, 16+ methods, manages full conversation lifecycle
- `Settings` (config.py) — pydantic-settings configuration from environment variables
- `ConversationState` (state.py) — state machine with transition validation

## Module Responsibilities
| Module | Responsibility |
|--------|---------------|
| `audio.py` | PyAudio microphone input, device validation |
| `vad.py` | Silero VAD for voice activity detection |
| `stt.py` | ElevenLabs Scribe v2 via WebSocket (persistent connection) |
| `llm.py` | Claude API integration, tool_use block detection |
| `tts.py` | ElevenLabs streaming TTS with pre-warmed connection pool |
| `playback.py` | Audio playback via paplay subprocess |
| `memory.py` | Conversation memory extraction |
| `graphrag.py` | Entity/relationship extraction for GraphRAG |
| `embeddings.py` | Voyage AI (voyage-4-lite) embedding generation |
| `context.py` | RAG context manager, assembles relevant context for LLM |
| `db.py` | asyncpg + pgvector (HNSW index) for vector storage |
| `app.py` | FastAPI endpoints, health check, WebSocket |
| `pipeline.py` | Main pipeline orchestrator |
| `config.py` | Settings via pydantic-settings |
| `state.py` | Conversation state machine |
| `processing.py` | Processing utilities |
| `persona.py` | Persona/character configuration |
| `log.py` | Structured logging |
| `logger.py` | Logger setup |
| `conversation_log.py` | Conversation history logging |

## Tool Loop
- Up to 3 iterations per LLM response
- Supports Home Assistant integration: `get_state`, `call_service`
- Supports web search tool
- SPEAKING -> PROCESSING transition enables multi-turn tool use

## Memory System
- GraphRAG with Haiku 4.5 for entity/relationship extraction
- Voyage voyage-4-lite embeddings
- PostgreSQL + pgvector with HNSW index
- Async database operations via asyncpg

## Source Layout
```
src/prot/          # 21 Python source files
tests/             # 19 test files
deploy/            # systemd service files
docs/plans/        # architectural decision documents
```
