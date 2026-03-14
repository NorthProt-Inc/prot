# SQLite Memory Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use execute-flow to implement this plan task-by-task.

**Goal:** One-shot migration script that reads SQLite conversation history, compacts via Gemini 3.1 Pro, extracts structured memories via Gemini 3 Flash, embeds via Voyage, and inserts into PostgreSQL.

**Architecture:** Single standalone async script with 4 sequential stages per day-chunk: compaction → extraction → embedding → DB insert. Each stage checkpoints to JSONL so re-runs skip completed work.

**Tech Stack:** `google-genai` (Gemini API), `voyageai` (embeddings), `asyncpg` + `pgvector` (PostgreSQL), `sqlite3` (stdlib)

---

### Task 1: Create migration script skeleton with config and checkpoint helpers

**Files:**
- Create: `scripts/migrate_memories.py`

**Step 1: Write the script skeleton**

Create `scripts/migrate_memories.py`:

```python
#!/usr/bin/env python3
"""One-shot migration: SQLite conversations → Gemini compaction → memory extraction → PostgreSQL.

Usage: python scripts/migrate_memories.py
"""

import asyncio
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

# ── API Keys (fill before running) ──────────────────────────────────
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
VOYAGE_API_KEY = "YOUR_VOYAGE_API_KEY"
DATABASE_URL = "postgresql://prot:prot@localhost:5432/prot"

# ── Model config ────────────────────────────────────────────────────
COMPACTION_MODEL = "gemini-3.1-pro-preview"
EXTRACTION_MODEL = "gemini-3-flash-preview"
VOYAGE_MODEL = "voyage-4-large"

# ── Paths ───────────────────────────────────────────────────────────
SQLITE_DB = Path("data/sqlite/sqlite_memory.db")
CHECKPOINT_DIR = Path("data/migration")

SUMMARIES_FILE = CHECKPOINT_DIR / "summaries.jsonl"
EXTRACTIONS_FILE = CHECKPOINT_DIR / "extractions.jsonl"
EMBEDDINGS_FILE = CHECKPOINT_DIR / "embeddings.jsonl"
INSERTED_FILE = CHECKPOINT_DIR / "inserted.jsonl"


# ── Checkpoint helpers ──────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict[str, dict]:
    """Load JSONL checkpoint file. Returns {date: data} dict."""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            result[entry["date"]] = entry
    return result


def append_checkpoint(path: Path, entry: dict) -> None:
    """Append a single JSON line to checkpoint file."""
    with path.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── SQLite reader ───────────────────────────────────────────────────

def load_messages_by_day() -> dict[str, list[dict]]:
    """Read all messages from SQLite, grouped by date (YYYY-MM-DD)."""
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT role, content, timestamp FROM messages ORDER BY timestamp"
    )
    by_day = defaultdict(list)
    for row in cursor:
        date = row["timestamp"][:10]
        by_day[date].append({
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
        })
    conn.close()
    return dict(sorted(by_day.items()))


def format_conversation(messages: list[dict]) -> str:
    """Format messages into a readable conversation transcript."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
```

**Step 2: Verify the skeleton parses**

Run: `python -c "import ast; ast.parse(open('scripts/migrate_memories.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/migrate_memories.py
git commit -m "feat: add migration script skeleton with config and checkpoint helpers"
```

---

### Task 2: Add Step 1 — Gemini compaction

**Files:**
- Modify: `scripts/migrate_memories.py`

**Step 1: Add prompts and compaction function**

Append after `format_conversation`:

```python
# ── Prompts ─────────────────────────────────────────────────────────

COMPACTION_PROMPT = (
    "Summarize the following day's conversation between Mark and Axel. "
    "Focus on: key topics discussed, decisions made, emotional highlights, "
    "and any behavioral patterns. Wrap your summary in <summary></summary> tags.\n"
    "Date: {date}"
)

EXTRACTION_PROMPT = """You are a memory extraction system. Given a conversation summary,
extract structured memories into 4 layers. The summary may be in Korean or English.
Keep names and terms in their original language.

Return JSON with this exact structure:
{
  "semantic": [
    {"category": "person|preference|fact|skill|relationship", "subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0}
  ],
  "episodic": {
    "summary": "...",
    "topics": ["..."],
    "emotional_tone": "warm|tense|playful|curious|neutral|...",
    "significance": 0.0-1.0,
    "duration_turns": 0
  },
  "emotional": [
    {"emotion": "joy|frustration|curiosity|gratitude|...", "trigger_context": "...", "intensity": 0.0-1.0}
  ],
  "procedural": [
    {"pattern": "...", "frequency": "daily|weekly|occasional|null", "confidence": 0.0-1.0}
  ]
}

Rules:
- semantic: SPO triples for facts, preferences, knowledge. Be specific and concise.
- episodic: ONE episode summarizing this conversation segment. Always include.
- emotional: Capture emotional highlights. Empty array if no notable emotions.
- procedural: Behavioral patterns observed. Empty array if none.
- If the summary has no meaningful content, return empty arrays and null episodic."""


# ── Step 1: Compaction ──────────────────────────────────────────────

async def run_compaction(days: dict[str, list[dict]]) -> dict[str, str]:
    """Compact each day's conversation via Gemini 3.1 Pro. Returns {date: summary}."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    checkpoint = load_checkpoint(SUMMARIES_FILE)
    results = {date: entry["summary"] for date, entry in checkpoint.items()}

    for date, messages in days.items():
        if date in results:
            print(f"  [compaction] {date} — cached")
            continue

        transcript = format_conversation(messages)
        prompt = COMPACTION_PROMPT.format(date=date)
        full_prompt = f"{transcript}\n\n{prompt}"

        print(f"  [compaction] {date} — {len(messages)} msgs, {len(transcript)} chars...")
        response = await client.aio.models.generate_content(
            model=COMPACTION_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=16384,
                thinking_config=types.ThinkingConfig(
                    thinking_level="high",
                ),
            ),
        )
        text = response.text or ""

        # Extract from <summary> tags
        if "<summary>" in text and "</summary>" in text:
            text = text.split("<summary>", 1)[1].split("</summary>", 1)[0]
        summary = text.strip()

        results[date] = summary
        append_checkpoint(SUMMARIES_FILE, {"date": date, "summary": summary})
        print(f"           → {len(summary)} chars summary")

    return results
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/migrate_memories.py').read()); print('OK')"`

**Step 3: Commit**

```bash
git add scripts/migrate_memories.py
git commit -m "feat: add compaction step using Gemini 3.1 Pro"
```

---

### Task 3: Add Step 2 — Gemini extraction

**Files:**
- Modify: `scripts/migrate_memories.py`

**Step 1: Add extraction function**

Append after `run_compaction`:

```python
# ── Step 2: Extraction ──────────────────────────────────────────────

def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses."""
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text


async def run_extraction(summaries: dict[str, str]) -> dict[str, dict]:
    """Extract structured 4-layer memories from each summary via Gemini 3 Flash."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    checkpoint = load_checkpoint(EXTRACTIONS_FILE)
    results = {date: entry["extraction"] for date, entry in checkpoint.items()}

    empty = {"semantic": [], "episodic": None, "emotional": [], "procedural": []}

    for date, summary in summaries.items():
        if date in results:
            print(f"  [extraction] {date} — cached")
            continue

        if not summary:
            results[date] = empty
            append_checkpoint(EXTRACTIONS_FILE, {"date": date, "extraction": empty})
            print(f"  [extraction] {date} — empty summary, skipped")
            continue

        print(f"  [extraction] {date} — {len(summary)} chars...")
        response = await client.aio.models.generate_content(
            model=EXTRACTION_MODEL,
            contents=summary,
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_PROMPT,
                temperature=1.0,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(
                    thinking_level="high",
                ),
            ),
        )
        raw = response.text or ""
        raw = strip_markdown_fences(raw)

        try:
            extraction = json.loads(raw)
        except json.JSONDecodeError:
            print(f"           ⚠ JSON parse failed, saving raw to checkpoint")
            extraction = empty

        results[date] = extraction
        append_checkpoint(EXTRACTIONS_FILE, {"date": date, "extraction": extraction})

        sem = len(extraction.get("semantic") or [])
        epi = bool(extraction.get("episodic"))
        emo = len(extraction.get("emotional") or [])
        proc = len(extraction.get("procedural") or [])
        print(f"           → sem={sem} epi={epi} emo={emo} proc={proc}")

    return results
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/migrate_memories.py').read()); print('OK')"`

**Step 3: Commit**

```bash
git add scripts/migrate_memories.py
git commit -m "feat: add extraction step using Gemini 3 Flash"
```

---

### Task 4: Add Step 3 — Voyage embedding

**Files:**
- Modify: `scripts/migrate_memories.py`

**Step 1: Add embedding function**

Append after `run_extraction`:

```python
# ── Step 3: Embedding ───────────────────────────────────────────────

def collect_texts_for_embedding(extraction: dict) -> list[str]:
    """Collect all texts that need embedding from an extraction result.

    Order matches memory.py save_extraction(): semantic, episodic, emotional, procedural.
    """
    texts = []
    for s in extraction.get("semantic") or []:
        texts.append(f"{s['subject']} {s['predicate']} {s['object']}")
    epi = extraction.get("episodic")
    if epi and epi.get("summary"):
        texts.append(epi["summary"])
    for e in extraction.get("emotional") or []:
        texts.append(f"{e['emotion']}: {e['trigger_context']}")
    for p in extraction.get("procedural") or []:
        texts.append(p["pattern"])
    return texts


async def run_embedding(extractions: dict[str, dict]) -> dict[str, list[list[float]]]:
    """Embed all extracted memory texts via Voyage AI."""
    import voyageai

    client = voyageai.AsyncClient(api_key=VOYAGE_API_KEY)
    checkpoint = load_checkpoint(EMBEDDINGS_FILE)
    results = {date: entry["embeddings"] for date, entry in checkpoint.items()}

    for date, extraction in extractions.items():
        if date in results:
            print(f"  [embedding] {date} — cached")
            continue

        texts = collect_texts_for_embedding(extraction)
        if not texts:
            results[date] = []
            append_checkpoint(EMBEDDINGS_FILE, {"date": date, "embeddings": []})
            print(f"  [embedding] {date} — no texts to embed")
            continue

        print(f"  [embedding] {date} — {len(texts)} texts...")
        result = await client.embed(
            texts=texts,
            model=VOYAGE_MODEL,
            input_type="document",
        )
        embeddings = result.embeddings

        results[date] = embeddings
        append_checkpoint(EMBEDDINGS_FILE, {"date": date, "embeddings": embeddings})
        print(f"           → {len(embeddings)} vectors ({len(embeddings[0])} dim)")

    return results
```

Note: `voyageai.AsyncClient` has no explicit close method — let it garbage-collect.

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/migrate_memories.py').read()); print('OK')"`

**Step 3: Commit**

```bash
git add scripts/migrate_memories.py
git commit -m "feat: add embedding step using Voyage AI"
```

---

### Task 5: Add Step 4 — PostgreSQL insert

**Files:**
- Modify: `scripts/migrate_memories.py`

**Step 1: Add DB insert function**

SQL patterns copied from `src/prot/graphrag.py`. Critical: must register pgvector codec.

Append after `run_embedding`:

```python
# ── Step 4: PostgreSQL insert ───────────────────────────────────────

async def run_insert(
    extractions: dict[str, dict],
    all_embeddings: dict[str, list[list[float]]],
) -> None:
    """Insert extracted memories with embeddings into PostgreSQL."""
    import asyncpg
    from pgvector.asyncpg import register_vector

    checkpoint = load_checkpoint(INSERTED_FILE)
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3, init=register_vector)

    try:
        for date in sorted(extractions):
            if date in checkpoint:
                print(f"  [insert] {date} — cached")
                continue

            extraction = extractions[date]
            embeddings = all_embeddings.get(date, [])
            semantics = extraction.get("semantic") or []
            episodic = extraction.get("episodic")
            emotionals = extraction.get("emotional") or []
            procedurals = extraction.get("procedural") or []

            if not semantics and not episodic and not emotionals and not procedurals:
                append_checkpoint(INSERTED_FILE, {"date": date, "ids": {}})
                print(f"  [insert] {date} — empty extraction, skipped")
                continue

            print(f"  [insert] {date}...")
            idx = 0
            ids = {"semantic": [], "episodic": None, "emotional": [], "procedural": []}

            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Semantic
                    for s in semantics:
                        emb = embeddings[idx] if idx < len(embeddings) else None
                        row = await conn.fetchrow(
                            """INSERT INTO semantic_memories
                               (category, subject, predicate, object, confidence, source, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (subject, predicate, object)
                            DO UPDATE SET mention_count = semantic_memories.mention_count + 1,
                                         confidence = GREATEST(semantic_memories.confidence, EXCLUDED.confidence),
                                         embedding = COALESCE(EXCLUDED.embedding, semantic_memories.embedding),
                                         updated_at = now()
                            RETURNING id""",
                            s["category"], s["subject"], s["predicate"], s["object"],
                            s.get("confidence", 1.0), "migration", emb,
                        )
                        ids["semantic"].append(str(row["id"]))
                        idx += 1

                    # Episodic
                    episode_id = None
                    if episodic and episodic.get("summary"):
                        emb = embeddings[idx] if idx < len(embeddings) else None
                        row = await conn.fetchrow(
                            """INSERT INTO episodic_memories
                               (summary, topics, emotional_tone, significance, duration_turns, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6) RETURNING id""",
                            episodic["summary"],
                            episodic.get("topics", []),
                            episodic.get("emotional_tone"),
                            episodic.get("significance", 0.5),
                            episodic.get("duration_turns", 0),
                            emb,
                        )
                        episode_id = row["id"]
                        ids["episodic"] = str(episode_id)
                        idx += 1

                    # Emotional (linked to episode)
                    for e in emotionals:
                        emb = embeddings[idx] if idx < len(embeddings) else None
                        row = await conn.fetchrow(
                            """INSERT INTO emotional_memories
                               (emotion, trigger_context, intensity, episode_id, embedding)
                            VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                            e["emotion"], e["trigger_context"],
                            e.get("intensity", 0.5), episode_id, emb,
                        )
                        ids["emotional"].append(str(row["id"]))
                        idx += 1

                    # Procedural
                    for p in procedurals:
                        emb = embeddings[idx] if idx < len(embeddings) else None
                        row = await conn.fetchrow(
                            """INSERT INTO procedural_memories
                               (pattern, frequency, confidence, embedding, last_observed)
                            VALUES ($1, $2, $3, $4, now())
                            ON CONFLICT (pattern)
                            DO UPDATE SET observation_count = procedural_memories.observation_count + 1,
                                         confidence = GREATEST(procedural_memories.confidence, EXCLUDED.confidence),
                                         frequency = COALESCE(EXCLUDED.frequency, procedural_memories.frequency),
                                         embedding = COALESCE(EXCLUDED.embedding, procedural_memories.embedding),
                                         last_observed = now(),
                                         updated_at = now()
                            RETURNING id""",
                            p["pattern"], p.get("frequency"),
                            p.get("confidence", 0.5), emb,
                        )
                        ids["procedural"].append(str(row["id"]))
                        idx += 1

            append_checkpoint(INSERTED_FILE, {"date": date, "ids": ids})
            sem_n = len(ids["semantic"])
            emo_n = len(ids["emotional"])
            proc_n = len(ids["procedural"])
            print(f"           → sem={sem_n} epi={bool(ids['episodic'])} emo={emo_n} proc={proc_n}")

    finally:
        await pool.close()
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/migrate_memories.py').read()); print('OK')"`

**Step 3: Commit**

```bash
git add scripts/migrate_memories.py
git commit -m "feat: add PostgreSQL insert step with pgvector support"
```

---

### Task 6: Add main() and wire everything together

**Files:**
- Modify: `scripts/migrate_memories.py`

**Step 1: Add main() at end of script**

```python
# ── Main ────────────────────────────────────────────────────────────

async def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading SQLite messages ===")
    days = load_messages_by_day()
    total_msgs = sum(len(msgs) for msgs in days.values())
    print(f"    {len(days)} days, {total_msgs} messages\n")

    print("=== Step 1: Compaction (Gemini 3.1 Pro) ===")
    summaries = await run_compaction(days)
    print(f"    {len(summaries)} summaries\n")

    print("=== Step 2: Extraction (Gemini 3 Flash) ===")
    extractions = await run_extraction(summaries)
    print(f"    {len(extractions)} extractions\n")

    print("=== Step 3: Embedding (Voyage AI) ===")
    embeddings = await run_embedding(extractions)
    non_empty = sum(1 for e in embeddings.values() if e)
    print(f"    {non_empty}/{len(embeddings)} days with embeddings\n")

    print("=== Step 4: PostgreSQL Insert ===")
    await run_insert(extractions, embeddings)
    print("\n=== Migration complete ===")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Full syntax check**

Run: `python -c "import ast; ast.parse(open('scripts/migrate_memories.py').read()); print('OK')"`

**Step 3: Commit**

```bash
git add scripts/migrate_memories.py
git commit -m "feat: wire up main() for complete migration pipeline"
```

---

### Task 7: Manual test run

**Not a code task — manual verification.**

**Step 1: Set API keys in the script**

Edit `scripts/migrate_memories.py` — fill `GOOGLE_API_KEY`, `VOYAGE_API_KEY`, `DATABASE_URL`.

**Step 2: Run the full pipeline**

```bash
python scripts/migrate_memories.py
```

**Step 3: Verify outputs**

- Check `data/migration/summaries.jsonl` — 25 lines, each with a `summary` field
- Check `data/migration/extractions.jsonl` — 25 lines, each with `extraction` containing 4 layers
- Check `data/migration/embeddings.jsonl` — 25 lines with vector arrays
- Check `data/migration/inserted.jsonl` — 25 lines with UUIDs

**Step 4: Verify PostgreSQL**

```bash
psql -U prot -d prot -c "SELECT COUNT(*) FROM semantic_memories WHERE source='migration';"
psql -U prot -d prot -c "SELECT COUNT(*) FROM episodic_memories;"
psql -U prot -d prot -c "SELECT COUNT(*) FROM emotional_memories;"
psql -U prot -d prot -c "SELECT COUNT(*) FROM procedural_memories;"
```

**Step 5: Test re-run (checkpoint verification)**

```bash
python scripts/migrate_memories.py
```

Expected: all steps print "cached" for every date. No API calls made.
