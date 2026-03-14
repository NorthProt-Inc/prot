# SQLite → PostgreSQL Memory Migration Design

## Goal

One-shot migration of conversation history from SQLite (`data/sqlite/sqlite_memory.db`)
into the 4-layer pgvector memory store. Uses Gemini models for compaction and extraction
to consume Google Cloud credits.

## Source Data

- **Database**: `data/sqlite/sqlite_memory.db`
- **Table**: `messages` (1,601 rows, ~985K chars)
- **Sessions**: 2 (Jan 7–28, Jan 28–Feb 1 2026)
- **Days**: 26 (2–190 msgs/day)
- **Roles**: `Mark` (user), `Axel` (assistant)

## Pipeline

```
SQLite messages (grouped by date, 26 chunks)
  │
  ├─ [Step 1] Compaction — gemini-3.1-pro-preview
  │    thinking: high, temp: 1.0, max_output: 16384
  │    → summaries.jsonl
  │
  ├─ [Step 2] Extraction — gemini-3-flash-preview
  │    thinking: high, temp: 1.0, max_output: 8192
  │    Reuses _EXTRACTION_PROMPT from memory.py
  │    → extractions.jsonl
  │
  ├─ [Step 3] Embedding — voyage-4-large (1024 dim)
  │    input_type="document", batch per day
  │    → embeddings.jsonl
  │
  └─ [Step 4] PostgreSQL upsert
       asyncpg direct (graphrag.py SQL patterns)
       → inserted.jsonl
```

## Models & Parameters

| Parameter         | Compaction (Step 1)       | Extraction (Step 2)       |
|-------------------|---------------------------|---------------------------|
| Model             | gemini-3.1-pro-preview    | gemini-3-flash-preview    |
| Thinking          | high                      | high                      |
| Temperature       | 1.0                       | 1.0                       |
| Max output tokens | 16,384                    | 8,192                     |

## Compaction Prompt

Custom prompt for day-level conversation summarization:

```
Summarize the following day's conversation between Mark and Axel.
Focus on: key topics discussed, decisions made, emotional highlights,
and any behavioral patterns. Wrap your summary in <summary></summary> tags.
Date: {date}
```

## Extraction Prompt

Reuses `_EXTRACTION_PROMPT` from `memory.py` — structured 4-layer JSON output
(semantic, episodic, emotional, procedural).

## Checkpoint Strategy

Directory: `data/migration/`

Each step writes a JSONL file. One line per date:
- `summaries.jsonl` — `{"date": "2026-01-07", "summary": "..."}`
- `extractions.jsonl` — `{"date": "2026-01-07", "extraction": {...}}`
- `embeddings.jsonl` — `{"date": "2026-01-07", "embeddings": [[...], ...]}`
- `inserted.jsonl` — `{"date": "2026-01-07", "ids": {"semantic": [...], ...}}`

On re-run, each step loads its checkpoint and skips already-processed dates.

## What Goes Into PostgreSQL

Only extracted memories (Step 2 output) with embeddings (Step 3).
Compaction summaries are NOT stored in DB — only in checkpoint files.

Target tables: `semantic_memories`, `episodic_memories`, `emotional_memories`,
`procedural_memories` (schema.sql).

## Script Details

- **Location**: `scripts/migrate_memories.py`
- **Standalone**: No prot package imports. Copies SQL patterns from graphrag.py.
- **API keys**: Hardcoded constants at top of script (GOOGLE_API_KEY, VOYAGE_API_KEY, DATABASE_URL).
- **Dependencies**: `google-genai`, `voyageai`, `asyncpg`

## Non-Goals

- No reranking (query-time feature, not relevant at insert time)
- No compaction results in DB (checkpoint files only)
- No changes to existing prot codebase
