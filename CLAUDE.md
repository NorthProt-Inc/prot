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
  state.py         # 6-state FSM (IDLE→LISTENING→PROCESSING→SPEAKING→ACTIVE→INTERRUPTED) + force_recovery()
  config.py        # Pydantic Settings — all env vars + LOCAL_TZ constant
  audio.py         # PyAudio microphone capture
  vad.py           # Silero VAD speech detection
  stt.py           # ElevenLabs Scribe v2 (WebSocket realtime STT)
  llm.py           # Claude API (Sonnet 4.6) — streaming responses + tool-use loop (max 3 rounds)
  hass.py          # Home Assistant conversation API delegation (HassAgent)
  tts.py           # ElevenLabs TTS streaming
  playback.py      # paplay (PulseAudio) audio output with producer-consumer queue
  processing.py    # Orchestrates LLM→TTS→playback per utterance
  context.py       # 3-block system prompt builder + conversation data container
  persona.py       # Axel persona definition (loaded from data/axel.xml)
  memory.py        # Compaction-driven 4-layer memory extraction + RAG context retrieval
  graphrag.py      # pgvector-backed 4-layer memory storage (semantic, episodic, emotional, procedural)
  decay.py         # AdaptiveDecayCalculator for time-decay memory scoring
  embeddings.py    # Voyage AI embeddings (voyage-4-large)
  reranker.py      # Voyage AI reranker (rerank-2.5)
  db.py            # asyncpg connection pool + schema init + CSV export on shutdown
  schema.sql       # PostgreSQL schema (auto-applied on startup)
  logging/         # Structured logging subsystem
    setup.py           # Logger configuration
    structured_logger.py  # StructuredLogger class
    formatters.py      # Log formatters
    handlers.py        # Log handlers
    constants.py       # Log field constants
    tracing.py         # Function tracing decorator with call-depth visualization
```

## Code Patterns

- **Async-first**: All pipeline stages are async. Use `asyncio.create_task` for background work.
- **State machine**: Transitions go through `StateMachine.transition()`. Never set state directly.
  Error recovery uses `StateMachine.force_recovery()` — the only sanctioned escape hatch (logs warning).
- **Barge-in**: During SPEAKING, VAD uses higher threshold (`VAD_THRESHOLD_SPEAKING=0.8`).
  Detection triggers INTERRUPTED state and TTS cancellation.
- **Tool loop**: LLM supports up to 3 tool-use rounds per response (Home Assistant, web search).
- **Prompt caching**: System prompt uses 3-block layout optimized for Anthropic cache hits.
  Block order matters — persona (static) → RAG context (semi-static) → dynamic.
- **Server-side compaction**: Beta API (`compact-2026-01-12` + `context-management-2025-06-27`)
  manages context automatically. Thinking clearing (keep 2 turns), tool result clearing (>30K tokens),
  and compaction (>50K tokens) all run server-side. No client-side trimming needed.
- **Memory extraction**: Compaction-driven. No per-exchange extraction.
  Fires on compaction events (pause_after_compaction) and shutdown (forced summarization).
  Haiku/Flash extracts 4-layer structured memories from compaction summary.
  Time-decay scoring (AdaptiveDecayCalculator) at query time.
  Embeddings: voyage-4-large. Reranker: rerank-2.5.
- **Reranker**: RAG retrieval uses Voyage rerank-2.5 to re-score candidate results before context injection.
  Improves relevance of memory recall without increasing embedding dimensionality.
- **DB optional**: App works without PostgreSQL — memory/RAG features gracefully degrade.
- **Beta API**: Uses `beta.messages.stream()` with compaction + context editing betas.
  Adaptive thinking + effort are GA features passed through beta endpoint.

## Testing

- Unit tests: `uv run pytest` (default, no API keys)
- Integration tests: `uv run pytest -m integration` (requires real API keys in .env)
- pytest-asyncio with `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio`
- Test files mirror source: `test_<module>.py` for each `<module>.py`

## Language Rules

- Conversation with Claude: Korean
- Code, comments, commit messages, docs: English
- Architectural decisions recorded in docs/ (when applicable)
