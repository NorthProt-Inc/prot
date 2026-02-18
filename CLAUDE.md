# prot

Real-time voice conversation system with Axel persona.
Streaming audio pipeline: Microphone → VAD → STT → LLM → TTS → Speaker.

## Commands

```bash
uv sync                              # Install dependencies
uv sync --extra dev                  # Install with dev deps
uv run pytest                        # Unit tests (no API keys needed)
uv run pytest -m integration         # Integration tests (requires .env API keys)
uv run pytest --cov=prot --cov-report=term-missing  # Coverage
uv run pytest tests/test_llm.py -v   # Single test file
./scripts/run.sh                     # Dev launcher (auto-kills stale port)
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --reload  # Manual dev server
```

## Architecture

```
src/prot/
  app.py           # FastAPI app, lifespan, HTTP endpoints (/health, /state, /diagnostics)
  pipeline.py      # Main voice pipeline orchestrator (run_pipeline)
  state.py         # 6-state FSM (IDLE→LISTENING→PROCESSING→SPEAKING→ACTIVE→INTERRUPTED)
  config.py        # Pydantic Settings — all env vars
  audio.py         # PyAudio microphone capture
  vad.py           # Silero VAD speech detection
  stt.py           # ElevenLabs Scribe v2 (WebSocket realtime STT)
  llm.py           # Claude API — streaming responses + tool-use loop (max 3 rounds)
  tts.py           # ElevenLabs TTS streaming
  playback.py      # paplay (PulseAudio) audio output with producer-consumer queue
  processing.py    # Orchestrates LLM→TTS→playback per utterance
  context.py       # 3-block system prompt builder (persona, RAG context, dynamic)
  persona.py       # Axel persona definition (loaded from docs/axel.json)
  memory.py        # Background memory extraction + RAG context retrieval
  graphrag.py      # Entity/relationship extraction via Haiku 4.5
  community.py     # Louvain community detection + LLM summarization
  embeddings.py    # Voyage AI embeddings (voyage-4-lite)
  db.py            # asyncpg connection pool + schema init + CSV export on shutdown
  conversation_log.py  # Daily JSONL conversation logger (data/conversations/)
  log.py           # Legacy logging compat
  schema.sql       # PostgreSQL schema (auto-applied on startup)
  logging/         # Structured logging subsystem (6 modules)
    setup.py           # Logger configuration
    structured_logger.py  # StructuredLogger class
    formatters.py      # Log formatters
    handlers.py        # Log handlers
    constants.py       # Log field constants
```

## Code Patterns

- **Async-first**: All pipeline stages are async. Use `asyncio.create_task` for background work.
- **State machine**: Transitions go through `StateMachine.transition()`. Never set state directly.
- **Barge-in**: During SPEAKING, VAD uses higher threshold (`VAD_THRESHOLD_SPEAKING=0.8`).
  Detection triggers INTERRUPTED state and TTS cancellation.
- **Tool loop**: LLM supports up to 3 tool-use rounds per response (Home Assistant, web search).
- **Prompt caching**: System prompt uses 3-block layout optimized for Anthropic cache hits.
  Block order matters — persona (static) → RAG context (semi-static) → dynamic.
- **Memory extraction**: Runs in background after each exchange. Uses Haiku 4.5 for entity/relationship extraction.
  Community detection triggers every 5 extractions via CommunityDetector (Louvain clustering).
- **DB optional**: App works without PostgreSQL — memory/RAG features gracefully degrade.

## Testing

- Unit tests: `uv run pytest` (default, no API keys)
- Integration tests: `uv run pytest -m integration` (requires real API keys in .env)
- pytest-asyncio with `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio`
- Test files mirror source: `test_<module>.py` for each `<module>.py`

## Language Rules

- Conversation with Claude: Korean
- Code, comments, commit messages, docs: English
- Architectural decisions recorded in `docs/plans/`
