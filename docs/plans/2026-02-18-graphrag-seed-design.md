# GraphRAG Seed Design

Seed PostgreSQL with entities, relationships, and communities from rewritten conversation pairs.

## Context

- **Input**: `data/migration/rewritten_pairs.jsonl` — 381 pairs (order 0–392)
- **Target**: Empty PostgreSQL tables (entities, relationships, communities, community_members)
- **Approach**: Reuse existing `MemoryExtractor` + `GraphRAGStore` + `AsyncVoyageEmbedder` + `CommunityDetector`
- **conversation_messages**: Deferred (message quality concerns)

## Change 1: Fix cross-extraction relationship loss

### Problem

`save_extraction` (memory.py:114–136) builds `entity_ids` only from the current extraction.
Relationships referencing entities from prior extractions are silently dropped:

```python
src_id = entity_ids.get(rel["source"])  # None if from prior extraction
tgt_id = entity_ids.get(rel["target"])  # None if from prior extraction
if src_id and tgt_id:                   # silently skipped
```

### Fix

Add `GraphRAGStore.get_entity_id_by_name(name, namespace, conn)` for DB fallback lookup.
In `save_extraction`, if source/target not in current `entity_ids`, look up by name in DB.

**graphrag.py** — new method:

```python
async def get_entity_id_by_name(
    self, name: str, namespace: str = "default", conn=None,
) -> UUID | None:
    query = "SELECT id FROM entities WHERE namespace = $1 AND name = $2"
    row = await (conn or self._pool).fetchrow(query, namespace, name)
    return row["id"] if row else None
```

**memory.py** — `save_extraction` relationship loop:

```python
for rel in relationships:
    src_id = entity_ids.get(rel["source"])
    tgt_id = entity_ids.get(rel["target"])
    if not src_id:
        src_id = await self._store.get_entity_id_by_name(rel["source"], conn=conn)
    if not tgt_id:
        tgt_id = await self._store.get_entity_id_by_name(rel["target"], conn=conn)
    if src_id and tgt_id:
        await self._store.upsert_relationship(...)
```

This fix benefits both real-time conversations and batch seeding.

## Change 2: Seed script

`scripts/seed_graphrag.py` — async script using existing modules.

### Data flow

```
rewritten_pairs.jsonl (381 pairs, chronological order)
    → accumulate as messages list
    → sliding window (last 6 messages = 3 turns)
    → MemoryExtractor.extract_from_conversation(window, known_entities)
    → MemoryExtractor.save_extraction(result)
    → known_entities auto-accumulate via _known_entities set
    → community rebuild every 5 extractions (automatic)
    → final community rebuild at end
```

### Script structure

```python
async def main():
    # 1. Load pairs from JSONL
    # 2. Init: DB pool → GraphRAGStore → AsyncVoyageEmbedder → CommunityDetector → MemoryExtractor
    # 3. Seed known entities from DB (for resume support)
    # 4. For each pair (in order):
    #    - Append user + assistant to messages list
    #    - Build window: last (memory_extraction_window_turns * 2) messages
    #    - known = sorted(extractor._known_entities)
    #    - extraction = await extractor.extract_from_conversation(window, known)
    #    - await extractor.save_extraction(extraction)
    #    - Log progress every 10 pairs
    #    - On error: log and continue
    # 5. Final community rebuild
    # 6. Print stats (entities, relationships, communities created)
    # 7. Cleanup: extractor.close(), embedder.close(), pool.close()
```

### CLI flags

- `--dry-run` — extract only, no DB writes (print extraction results)
- `--limit N` — process first N pairs only
- `--start-from N` — resume from order N
- `--delay SECONDS` — delay between pairs (rate limit, default 0.5)

### Dependencies

Existing modules only:
- `prot.db.init_pool`
- `prot.graphrag.GraphRAGStore`
- `prot.embeddings.AsyncVoyageEmbedder`
- `prot.community.CommunityDetector`
- `prot.memory.MemoryExtractor`
- `prot.config.settings`

### Prerequisites

- `.env` must have: `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY` (Settings requirement), `VOYAGE_API_KEY`, `DATABASE_URL`
- PostgreSQL running with schema applied
- `data/migration/rewritten_pairs.jsonl` exists

### Resource cleanup

All resources closed in `finally` block:
- `await extractor.close()` — Anthropic client + community detector + reranker
- `await embedder.close()` — Voyage client
- `await pool.close()` — DB pool

## Verification

```bash
# Dry run — check extraction quality
uv run python scripts/seed_graphrag.py --dry-run --limit 3

# Small test — 10 pairs into DB
uv run python scripts/seed_graphrag.py --limit 10

# Check DB state
PGPASSWORD=prot psql -h localhost -U prot -d prot -c "
SELECT 'entities' as tbl, count(*) FROM entities
UNION ALL SELECT 'relationships', count(*) FROM relationships
UNION ALL SELECT 'communities', count(*) FROM communities;"

# Full run
uv run python scripts/seed_graphrag.py --delay 0.5

# Verify community coverage
PGPASSWORD=prot psql -h localhost -U prot -d prot -c "
SELECT c.id, c.entity_count, LEFT(c.summary, 80)
FROM communities c ORDER BY entity_count DESC;"
```
