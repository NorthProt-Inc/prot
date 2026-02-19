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
  llm.py           # Claude API (Sonnet 4.6) — streaming responses + tool-use loop (max 3 rounds)
  tts.py           # ElevenLabs TTS streaming
  playback.py      # paplay (PulseAudio) audio output with producer-consumer queue
  processing.py    # Orchestrates LLM→TTS→playback per utterance
  context.py       # 3-block system prompt builder + sliding window context (last N turns)
  persona.py       # Axel persona definition (loaded from docs/axel.json)
  memory.py        # Background memory extraction + RAG context retrieval
  graphrag.py      # Entity/relationship extraction via Haiku 4.5
  community.py     # Louvain community detection + LLM summarization
  embeddings.py    # Voyage AI embeddings (voyage-4, voyage-context-3 contextual)
  reranker.py      # Voyage AI reranker (rerank-2.5)
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
- **Sliding window**: `get_recent_messages(max_turns)` returns last N turns for LLM calls.
  Full history preserved via `get_messages()` for session logging and memory extraction.
  Window boundary skips orphaned tool_result messages to maintain valid conversation flow.
- **Memory extraction**: Runs in background after each exchange. Uses Haiku 4.5 for entity/relationship extraction.
  Contextual embeddings (voyage-context-3) for document-aware entity storage. Reranker (rerank-2.5) refines RAG results.
  Community detection triggers every 5 extractions via CommunityDetector (Louvain clustering).
- **Reranker**: RAG retrieval uses Voyage rerank-2.5 to re-score candidate results before context injection.
  Improves relevance of memory recall without increasing embedding dimensionality.
- **DB optional**: App works without PostgreSQL — memory/RAG features gracefully degrade.
- **GA API**: Uses `messages.stream()` (not beta). Adaptive thinking + effort are GA on Sonnet 4.6.

## Testing

- Unit tests: `uv run pytest` (default, no API keys)
- Integration tests: `uv run pytest -m integration` (requires real API keys in .env)
- pytest-asyncio with `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio`
- Test files mirror source: `test_<module>.py` for each `<module>.py`

## Language Rules

- Conversation with Claude: Korean
- Code, comments, commit messages, docs: English
- Architectural decisions recorded in `docs/plans/`
