# Voice Architecture Design — prot

> Real-time voice conversation system with Axel persona

## Architecture: Interruptible Streaming Pipeline (Approach B)

Half-duplex streaming with barge-in support. Single FastAPI process, systemd managed.

## Components

```
prot.service (FastAPI)
├── AudioManager      — PyAudio mic input / paplay output
├── VADProcessor      — Silero VAD (always-on when mic active)
├── STTClient         — Deepgram Flux WebSocket streaming
├── LLMClient         — Claude API streaming + tool execution
├── TTSClient         — ElevenLabs Flash v2.5 streaming
└── ContextManager    — prompt cache, GraphRAG, persona, memory
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | FastAPI | Async-first, single process |
| STT | Deepgram Flux (WebSocket) | Built-in turn detection (~260ms), Korean support |
| LLM | Claude Opus 4.6 (`claude-opus-4-6`) | Adaptive thinking, effort parameter, prompt caching |
| TTS | ElevenLabs Flash v2.5 (`eleven_flash_v2_5`) | 75ms TTFB, 50% cheaper, Korean support |
| Search | Anthropic `web_search_20250305` | Server-side, no MCP overhead |
| IoT | Home Assistant (native tool, REST API) | Direct FastAPI execution, no MCP |
| Vector DB | pgvector + PostgreSQL | Familiar, handles vectors + relational |
| Embeddings | voyage-3.5-lite (1024dim) | Default dimension, $0.03/MTok, Matryoshka |
| VAD | Silero VAD (~1MB, local) | Always-on gate, $0 cost |
| Audio output | paplay (PulseAudio) | Simple, stable, async subprocess |
| Deployment | Ubuntu local server, systemd | SSH access, single user |

## State Machine

```
IDLE → LISTENING → PROCESSING → SPEAKING → ACTIVE → IDLE
                                    ↓
                              INTERRUPTED → LISTENING
```

- **IDLE**: VAD only ($0). Mic ON = conversation mode.
- **LISTENING**: VAD speech → Deepgram WS open. Flux turn detection.
- **PROCESSING**: Claude streaming + tool execution.
- **SPEAKING**: TTS + paplay. VAD still active (elevated threshold).
- **INTERRUPTED**: User speaks during TTS → kill paplay, flush TTS, cancel LLM → LISTENING.
- **ACTIVE**: TTS done, 30s keep-alive on Deepgram WS for quick ping-pong.

## Streaming Pipeline

All stages overlap. Next stage starts before previous completes.

```
Mic PCM (32ms chunks)
  → Silero VAD (speech gate)
    → Deepgram Flux WS (interim → final, ~260ms turn detection)
      → Claude streaming (adaptive thinking, medium effort)
        → Sentence chunking + sanitize_for_tts()
          → ElevenLabs Flash streaming (pcm_16000)
            → paplay stdin pipe
```

### Latency Budget

| Stage | Expected |
|-------|----------|
| VAD detection | ~96ms |
| Deepgram Flux turn detection | ~260ms |
| Claude TTFB (cache hit) | ~300-500ms |
| Sentence accumulation | ~200-400ms |
| ElevenLabs Flash TTFB | ~75ms |
| **Total to first audio** | **~0.9-1.3s** |

## Prompt Architecture (4 Cache Breakpoints)

```
Block 1: Persona + Rules (~800 tok)       → cache_control: ephemeral  [STATIC]
Block 2: GraphRAG Context (~3,000+ tok)   → cache_control: ephemeral  [TOPIC-DEPENDENT]
Block 3: Dynamic Context (~200 tok)       → NO cache_control          [PER-REQUEST, LAST]
Tools: web_search, home_assistant         → cache on last tool         [STATIC]
Messages: conversation history
```

> **Why this order matters:** cache_control: ephemeral creates a cache breakpoint at that
> position. Cached prefix must be identical across requests. If a per-request block (datetime)
> sits between two cached blocks, it invalidates all downstream cache — Block 2 (GraphRAG)
> would NEVER hit cache. Placing all dynamic content LAST preserves the cached prefix.

### LLM Configuration

```python
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1500,
    thinking={"type": "adaptive"},
    effort="medium",               # Fixed for casual conversation
    stream=True,
    system=[cached_persona, cached_rag, dynamic_context],
    tools=[web_search_tool, hass_tool],
    messages=conversation_history,
)
```

### Cache Strategy

**TTL: 5m (default — no explicit configuration needed)**
- During conversation: turns every few seconds → timer auto-refreshes → always warm
- Between conversations: cache miss is acceptable (new GraphRAG context anyway)
- Cost: 1.25x write vs 2x for 1h — not worth the premium for casual conversation
- Syntax: `{"type": "ephemeral"}` (no ttl field needed, 5m is default)

**Minimum Cache Tokens: 4,096 (Opus 4.6)**
- tools(~500) + Block 1(~800) = ~1,300 → under minimum, persona alone NOT cacheable
- tools(~500) + Block 1(~800) + Block 2 GraphRAG(~3,000+) = ~4,300+ → exceeds minimum
- Strategy: always load ≥3,000 tokens of GraphRAG context into Block 2
- Existing SQLite conversations → pre-populate GraphRAG → cache hits from first conversation

**Cache Behavior:**
- No duplicate storage: identical prefix = cache READ (0.1x), not new WRITE
- TTL-based eviction: each cached prefix has independent timer, no size limit
- Timer resets on every hit: within conversation, effectively infinite cache lifetime
- Cross-conversation: Block 2 changes (different topic) → new write, but Block 1 prefix too short to cache independently

**Pricing (Opus 4.6):**

| Type | Cost/MTok |
|------|-----------|
| Base input | $5.00 |
| 5m cache write | $6.25 (1.25x) |
| Cache read | $0.50 (0.1x) |

### Effort: Fixed Medium

All conversation treated as medium effort (max_tokens=1500). Casual daily conversation does not need dynamic routing. Revisit if use cases expand.

## Persona

Static Axel persona (~800 tokens). Voice-optimized with:
- TTS-safe output rules (no markdown, emoji, bullets, numbered lists)
- Ping-pong rhythm (1-3 sentences, match user energy)
- Cynical Tech Bro voice style
- Korean with tech metaphors

Source: `/home/cyan/workplace/prot/axel.md`

## Audio I/O

### Input
- PyAudio callback mode, 16kHz mono, 512 samples/chunk (32ms)
- Device index configurable via env

### Output
- paplay subprocess: `--format=s16le --rate=16000 --channels=1`
- Async pipe, `process.kill()` for barge-in

### Barge-in
1. SPEAKING state: VAD threshold elevated (0.5 → 0.8)
2. User speech detected → paplay.kill() → TTS flush → LLM cancel
3. VAD threshold restored → LISTENING state

## Memory & Context

### Database Schema (pgvector + PostgreSQL)

**Tables:**

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `entities` | UUID PK, namespace, name, entity_type, description, attributes (JSONB), name_embedding vector(1024), mention_count | Named entities (people, places, topics) |
| `relationships` | source_id FK, target_id FK, relation_type, description, weight, attributes (JSONB) | Entity-to-entity edges |
| `communities` | level (hierarchical Leiden), summary, summary_embedding vector(1024), entity_count | Community detection clusters |
| `community_members` | community_id, entity_id | Junction table |
| `conversation_messages` | conversation_id, role, content, content_embedding vector(1024), metadata (JSONB) | Searchable conversation history |

**Indexes:**
- HNSW on all vector(1024) columns: `WITH (m = 16, ef_construction = 128)`
- pg_trgm GIN on entity names for fuzzy matching
- Composite indexes on (namespace, entity_type), (source_id, target_id)

**Connection:** asyncpg pool (min=2, max=10)

### Embedding Pipeline

- Model: voyage-3.5-lite @ 1024 dimensions (default)
- Cost: ~$0.03/MTok, ~52ms latency
- Async client: `voyageai.AsyncClient()`
- input_type: `"query"` for search, `"document"` for storage
- Batch: max 128 texts per call, semaphore for concurrency control

### Memory Lifecycle (Detailed)

```
CONVERSATION ENDS (30s idle → IDLE)
  → Background extraction task:
    1. Collect messages from this conversation session
    2. Send to Claude Haiku 4.5 for entity/relationship extraction (structured JSON output)
    3. Embed entity descriptions with voyage-3.5-lite (input_type="document")
    4. Upsert entities + relationships into pgvector
    5. Log: extraction metrics (entities found, relationships, latency)

CONVERSATION STARTS (IDLE → LISTENING)
  → Pre-load task:
    1. Embed recent user context with voyage-3.5-lite (input_type="query")
    2. Search communities by cosine similarity (top-k, k tunable, target ~3,000 tokens)
    3. Assemble summaries into Block 2 text
    4. ContextManager.update_rag_context(assembled_text)
```

### Existing Data Migration

- Source: genuine-axel-main SQLite (`backend/memory/recent/`)
- Existing cleanup scripts in `genuine-axel-main/scripts/` (memory_gc, dedup_knowledge_graph, etc.)
- Migration plan: clean SQLite → extract entities/relationships → populate pgvector
- This enables cache hits from first conversation

### Conversation Logs

- Daily JSON files (archival, NOT in LLM context) — kept as-is
- DB `conversation_messages` table for semantic search and extraction source

### Context Management
- Compaction: `compact_20260112` for long conversations
- Context Editing: clear old tool results (`clear_tool_uses_20250919`)

## Tools

### web_search (Server-side)
- Anthropic built-in `web_search_20250305`
- max_uses: 1 (latency constraint)
- user_location: Seoul, KR

### home_assistant (Native)
- FastAPI directly calls HASS REST API (httpx)
- Actions: get_state, call_service
- Provides: time, location, weather, device control

### Current Time
- Injected in Block 3 (non-cached, last) every request
- `datetime.now().strftime('%Y-%m-%d %H:%M:%S')` + timezone

## STT Correction

Two-layer, no separate LLM:
1. Deepgram `keyterm` parameter (up to 100 terms, 6x recognition boost)
2. System prompt instruction: interpret STT errors via conversation context

## Connection Lifecycle

```
IDLE ($0/hr)          — Silero VAD only
LISTENING ($0.46/hr)  — Deepgram WS active
ACTIVE (30s)          — WS keep-alive between turns
IDLE                  — 30s silence → WS close
```

## Key API Notes (Opus 4.6)

- Model ID: `claude-opus-4-6` (no date suffix)
- Prefill: REMOVED (400 error). Use system prompt for format control.
- `output_format` → `output_config.format` (deprecated)
- `thinking: {type: "enabled"}` + `budget_tokens` → deprecated. Use adaptive + effort.
- Fine-grained tool streaming: GA (no beta header)
- Max output: 128K tokens

## Post-Processing

```python
def sanitize_for_tts(text: str) -> str:
    text = re.sub(r'[*_#`~\[\](){}|>]', '', text)  # markdown
    text = re.sub(r'\d+\.\s', '', text)              # numbered lists
    text = re.sub(r'[-•]\s', '', text)                # bullets
    return text.strip()

def ensure_complete_sentence(text: str) -> str:
    for i in range(len(text)-1, -1, -1):
        if text[i] in '.!?~':
            return text[:i+1]
    return text
```

## Dependencies

**Core pipeline:**
- anthropic, deepgram-sdk, elevenlabs, pyaudio, silero-vad, httpx, fastapi, uvicorn, pydantic-settings

**Memory/GraphRAG (NEW):**
- asyncpg, voyageai, numpy, pgvector
