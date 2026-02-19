"""GraphRAG database garbage collection and deduplication.

Usage:
    uv run python scripts/db_gc.py check                       # Read-only diagnostic report
    uv run python scripts/db_gc.py full --dry-run               # Preview all changes
    uv run python scripts/db_gc.py full --apply                 # Execute all phases
    uv run python scripts/db_gc.py full --apply --skip-rebuild  # Skip community rebuild
    uv run python scripts/db_gc.py full --apply --skip-vacuum   # Skip VACUUM
    uv run python scripts/db_gc.py phase entity-dedup --dry-run # Single phase
"""

from __future__ import annotations

import argparse
import asyncio
import re
import unicodedata

import asyncpg

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

CANONICAL_RELATION_TYPES = frozenset({
    "contains", "part_of", "instance_of",
    "causes", "enables", "precedes", "triggers",
    "requires", "depends_on", "supports",
    "related_to", "similar_to", "contrasts_with", "alternative_to",
    "works_at", "works_with", "knows", "created_by", "owns",
    "uses", "produces", "improves", "replaces",
    "located_in",
    "prefers", "interested_in",
})

_RELATION_MAP: dict[str, str] = {
    # Korean → canonical
    "원인": "causes",
    "유발": "causes",
    "영향": "causes",
    "연관": "related_to",
    "관련": "related_to",
    "연결": "related_to",
    "관계": "related_to",
    "계획": "related_to",
    "상태": "related_to",
    "대상": "related_to",
    "목표": "related_to",
    "사용": "uses",
    "활용": "uses",
    "이용": "uses",
    "적용": "uses",
    "포함": "contains",
    "구성": "contains",
    "구성요소": "contains",
    "결과": "causes",
    "해결책": "improves",
    "해결": "improves",
    "실행": "triggers",
    "수행": "triggers",
    "트리거": "triggers",
    "발생": "triggers",
    "촉발": "triggers",
    "대체": "replaces",
    "선호": "prefers",
    "대안": "alternative_to",
    "비유": "similar_to",
    "선행조건": "precedes",
    "선행": "precedes",
    "구현": "produces",
    "요청": "requires",
    # English variants
    "caused": "causes",
    "caused_by": "causes",
    "affects": "causes",
    "includes": "contains",
    "utilizes": "uses",
    "resolves": "improves",
    "replacement": "replaces",
    "option": "alternative_to",
    "implements": "produces",
    "plans": "related_to",
}


def normalize_name(name: str) -> str:
    """Normalize entity name for dedup comparison.

    NFC unicode → lowercase → replace _/-/() with spaces → collapse whitespace.
    """
    name = unicodedata.normalize("NFC", name)
    name = name.lower()
    name = re.sub(r"[_\-\(\)/]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalize_relation_type(raw: str) -> str:
    """Map a raw relation type string to one of the canonical types."""
    cleaned = raw.strip().lower().replace("-", "_").replace(" ", "_")
    # Already canonical?
    if cleaned in CANONICAL_RELATION_TYPES:
        return cleaned
    # In the explicit map?
    if cleaned in _RELATION_MAP:
        return _RELATION_MAP[cleaned]
    # Try original (pre-cleaned) in map (for Korean chars)
    stripped = raw.strip()
    if stripped in _RELATION_MAP:
        return _RELATION_MAP[stripped]
    return "related_to"


def dedup_description(desc: str) -> str:
    """Remove duplicate lines from a newline-separated description."""
    if "\n" not in desc:
        return desc
    return "\n".join(dict.fromkeys(desc.split("\n")))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _connect() -> asyncpg.Connection:
    """Connect to PostgreSQL using DSN from prot.config.settings."""
    # Import here so pure functions are testable without prot installed
    from prot.config import settings  # noqa: PLC0415
    return await asyncpg.connect(settings.database_url)


# ---------------------------------------------------------------------------
# Check (read-only diagnostics)
# ---------------------------------------------------------------------------

async def run_check(conn: asyncpg.Connection) -> None:
    """Print a diagnostic report without modifying anything."""
    print("=" * 60)
    print("GraphRAG Database Health Check")
    print("=" * 60)

    # Entity stats
    total = await conn.fetchval("SELECT count(*) FROM entities")
    print(f"\nEntities: {total}")
    rows = await conn.fetch(
        "SELECT entity_type, count(*) AS cnt FROM entities GROUP BY 1 ORDER BY 2 DESC"
    )
    for r in rows:
        pct = r["cnt"] / total * 100 if total else 0
        print(f"  {r['entity_type']}: {r['cnt']} ({pct:.1f}%)")

    # Duplicate candidates (normalized exact match)
    all_entities = await conn.fetch("SELECT id, name, entity_type, mention_count FROM entities")
    groups: dict[str, list] = {}
    for e in all_entities:
        key = normalize_name(e["name"])
        groups.setdefault(key, []).append(e)
    dupes = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"\nDuplicate candidates (normalized exact): {len(dupes)} groups")
    for key, members in sorted(dupes.items(), key=lambda x: -len(x[1]))[:20]:
        names = [m["name"] for m in members]
        print(f"  '{key}' → {names}")

    # Trigram candidates
    trgm_rows = await conn.fetch("""
        SELECT a.id AS id_a, a.name AS name_a, b.id AS id_b, b.name AS name_b,
               similarity(a.name, b.name) AS sim
        FROM entities a, entities b
        WHERE a.id < b.id AND similarity(a.name, b.name) > 0.7
        ORDER BY sim DESC LIMIT 30
    """)
    print(f"\nTrigram similar pairs (>0.7): {len(trgm_rows)}")
    for r in trgm_rows[:20]:
        print(f"  {r['name_a']} ↔ {r['name_b']}  (sim={r['sim']:.3f})")

    # Relationship stats
    rel_total = await conn.fetchval("SELECT count(*) FROM relationships")
    rel_types = await conn.fetchval("SELECT count(DISTINCT relation_type) FROM relationships")
    print(f"\nRelationships: {rel_total}  |  Unique types: {rel_types}")
    type_rows = await conn.fetch(
        "SELECT relation_type, count(*) AS cnt FROM relationships GROUP BY 1 ORDER BY 2 DESC LIMIT 30"
    )
    for r in type_rows:
        canonical = normalize_relation_type(r["relation_type"])
        tag = "" if r["relation_type"] == canonical else f" → {canonical}"
        print(f"  {r['relation_type']}: {r['cnt']}{tag}")

    # Orphans
    orphan_count = await conn.fetchval("""
        SELECT count(*) FROM entities e
        LEFT JOIN relationships r ON r.source_id = e.id OR r.target_id = e.id
        WHERE r.id IS NULL
    """)
    print(f"\nOrphan entities (no relationships): {orphan_count}")

    # Description bloat
    desc_stats = await conn.fetchrow("""
        SELECT avg(length(description))::int AS avg_len,
               max(length(description)) AS max_len,
               count(*) FILTER (WHERE description LIKE '%\n%') AS multiline_count,
               count(*) FILTER (WHERE length(description) > 400) AS over_400
        FROM entities
    """)
    print(f"\nDescription stats:")
    print(f"  Avg length: {desc_stats['avg_len']}")
    print(f"  Max length: {desc_stats['max_len']}")
    print(f"  Multiline: {desc_stats['multiline_count']}")
    print(f"  Over 400 chars: {desc_stats['over_400']}")

    # Communities
    comm_count = await conn.fetchval("SELECT count(*) FROM communities")
    print(f"\nCommunities: {comm_count}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Phase 1: Entity dedup
# ---------------------------------------------------------------------------

async def phase_entity_dedup(conn: asyncpg.Connection, *, apply: bool) -> None:
    """Deduplicate entities by normalized name match + optional trigram."""
    print("\n--- Phase 1: Entity Dedup ---")

    all_entities = await conn.fetch(
        "SELECT id, name, entity_type, mention_count, created_at FROM entities ORDER BY mention_count DESC"
    )
    # Group by normalized name
    groups: dict[str, list] = {}
    for e in all_entities:
        key = normalize_name(e["name"])
        groups.setdefault(key, []).append(e)

    dupes = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"  Duplicate groups (normalized exact): {len(dupes)}")

    merge_count = 0
    skip_count = 0

    for key, members in dupes.items():
        # Check type mismatch
        types = {m["entity_type"] for m in members}
        if len(types) > 1:
            names = [m["name"] for m in members]
            print(f"  SKIP (type mismatch): {names} types={types}")
            skip_count += 1
            continue

        # Sort: highest mention_count first, then oldest created_at
        members.sort(key=lambda m: (-m["mention_count"], m["created_at"]))
        keeper = members[0]
        duplicates = members[1:]

        for dup in duplicates:
            print(f"  MERGE: '{dup['name']}' → '{keeper['name']}'")
            if apply:
                await _merge_entity(conn, keeper_id=keeper["id"], dup_id=dup["id"])
            merge_count += 1

    # Trigram pass for remaining near-duplicates
    trgm_rows = await conn.fetch("""
        SELECT a.id AS id_a, a.name AS name_a, a.entity_type AS type_a,
               a.mention_count AS mc_a, a.created_at AS ca_a,
               b.id AS id_b, b.name AS name_b, b.entity_type AS type_b,
               b.mention_count AS mc_b, b.created_at AS ca_b,
               similarity(a.name, b.name) AS sim
        FROM entities a, entities b
        WHERE a.id < b.id AND similarity(a.name, b.name) > 0.7
        ORDER BY sim DESC
    """)
    # Filter out pairs already merged by normalized match
    merged_ids: set = set()
    for r in trgm_rows:
        if r["id_a"] in merged_ids or r["id_b"] in merged_ids:
            continue
        if normalize_name(r["name_a"]) == normalize_name(r["name_b"]):
            continue  # Already handled above
        if r["type_a"] != r["type_b"]:
            print(f"  SKIP trigram (type mismatch): '{r['name_a']}' ↔ '{r['name_b']}' sim={r['sim']:.3f}")
            skip_count += 1
            continue
        # Pick keeper by mention_count
        if r["mc_a"] >= r["mc_b"]:
            keeper_id, keeper_name = r["id_a"], r["name_a"]
            dup_id, dup_name = r["id_b"], r["name_b"]
        else:
            keeper_id, keeper_name = r["id_b"], r["name_b"]
            dup_id, dup_name = r["id_a"], r["name_a"]

        print(f"  MERGE (trigram {r['sim']:.3f}): '{dup_name}' → '{keeper_name}'")
        if apply:
            await _merge_entity(conn, keeper_id=keeper_id, dup_id=dup_id)
        merged_ids.add(dup_id)
        merge_count += 1

    print(f"  Total merges: {merge_count}, skipped: {skip_count}")


async def _merge_entity(conn: asyncpg.Connection, *, keeper_id, dup_id) -> None:
    """Merge duplicate entity into keeper within a transaction."""
    async with conn.transaction():
        # Reassign relationships: source_id
        await conn.execute("""
            UPDATE relationships SET source_id = $1
            WHERE source_id = $2
            AND NOT EXISTS (
                SELECT 1 FROM relationships r2
                WHERE r2.source_id = $1 AND r2.target_id = relationships.target_id
                AND r2.relation_type = relationships.relation_type
            )
        """, keeper_id, dup_id)
        # For conflicting source_id rows: sum weights then delete
        conflicting_src = await conn.fetch("""
            SELECT r_dup.id AS dup_rel_id, r_dup.weight AS dup_weight,
                   r_keep.id AS keep_rel_id
            FROM relationships r_dup
            JOIN relationships r_keep
              ON r_keep.source_id = $1
             AND r_keep.target_id = r_dup.target_id
             AND r_keep.relation_type = r_dup.relation_type
            WHERE r_dup.source_id = $2
        """, keeper_id, dup_id)
        for row in conflicting_src:
            await conn.execute(
                "UPDATE relationships SET weight = weight + $1 WHERE id = $2",
                row["dup_weight"], row["keep_rel_id"],
            )
            await conn.execute("DELETE FROM relationships WHERE id = $1", row["dup_rel_id"])

        # Reassign relationships: target_id
        await conn.execute("""
            UPDATE relationships SET target_id = $1
            WHERE target_id = $2
            AND NOT EXISTS (
                SELECT 1 FROM relationships r2
                WHERE r2.source_id = relationships.source_id AND r2.target_id = $1
                AND r2.relation_type = relationships.relation_type
            )
        """, keeper_id, dup_id)
        # For conflicting target_id rows
        conflicting_tgt = await conn.fetch("""
            SELECT r_dup.id AS dup_rel_id, r_dup.weight AS dup_weight,
                   r_keep.id AS keep_rel_id
            FROM relationships r_dup
            JOIN relationships r_keep
              ON r_keep.source_id = r_dup.source_id
             AND r_keep.target_id = $1
             AND r_keep.relation_type = r_dup.relation_type
            WHERE r_dup.target_id = $2
        """, keeper_id, dup_id)
        for row in conflicting_tgt:
            await conn.execute(
                "UPDATE relationships SET weight = weight + $1 WHERE id = $2",
                row["dup_weight"], row["keep_rel_id"],
            )
            await conn.execute("DELETE FROM relationships WHERE id = $1", row["dup_rel_id"])

        # Remove self-references
        await conn.execute(
            "DELETE FROM relationships WHERE source_id = $1 AND target_id = $1", keeper_id,
        )

        # Sum mention counts
        dup_mc = await conn.fetchval("SELECT mention_count FROM entities WHERE id = $1", dup_id)
        if dup_mc:
            await conn.execute(
                "UPDATE entities SET mention_count = mention_count + $1 WHERE id = $2",
                dup_mc, keeper_id,
            )

        # Delete duplicate entity (CASCADE handles remaining FK refs)
        await conn.execute("DELETE FROM entities WHERE id = $1", dup_id)


# ---------------------------------------------------------------------------
# Phase 2: Relation type normalization
# ---------------------------------------------------------------------------

async def phase_relation_norm(conn: asyncpg.Connection, *, apply: bool) -> None:
    """Normalize relation types to canonical set."""
    print("\n--- Phase 2: Relation Type Normalization ---")

    type_rows = await conn.fetch(
        "SELECT relation_type, count(*) AS cnt FROM relationships GROUP BY 1 ORDER BY 2 DESC"
    )

    remap: dict[str, str] = {}
    for r in type_rows:
        old = r["relation_type"]
        new = normalize_relation_type(old)
        if old != new:
            remap[old] = new

    print(f"  Types to remap: {len(remap)} (of {len(type_rows)} total)")
    for old, new in sorted(remap.items()):
        print(f"    {old} → {new}")

    if not apply:
        return

    async with conn.transaction():
        for old_type, new_type in remap.items():
            # First: update non-conflicting rows
            await conn.execute("""
                UPDATE relationships SET relation_type = $1
                WHERE relation_type = $2
                AND NOT EXISTS (
                    SELECT 1 FROM relationships r2
                    WHERE r2.source_id = relationships.source_id
                    AND r2.target_id = relationships.target_id
                    AND r2.relation_type = $1
                )
            """, new_type, old_type)

            # Handle conflicts: sum weights into existing, delete conflicting
            conflicting = await conn.fetch("""
                SELECT r_old.id AS old_id, r_old.weight AS old_weight,
                       r_new.id AS new_id
                FROM relationships r_old
                JOIN relationships r_new
                  ON r_new.source_id = r_old.source_id
                 AND r_new.target_id = r_old.target_id
                 AND r_new.relation_type = $1
                WHERE r_old.relation_type = $2
            """, new_type, old_type)
            for row in conflicting:
                await conn.execute(
                    "UPDATE relationships SET weight = weight + $1 WHERE id = $2",
                    row["old_weight"], row["new_id"],
                )
                await conn.execute("DELETE FROM relationships WHERE id = $1", row["old_id"])

    remaining = await conn.fetchval("SELECT count(DISTINCT relation_type) FROM relationships")
    print(f"  Unique types after normalization: {remaining}")


# ---------------------------------------------------------------------------
# Phase 3: Description dedup
# ---------------------------------------------------------------------------

async def phase_desc_dedup(conn: asyncpg.Connection, *, apply: bool) -> None:
    """Remove duplicate lines within entity descriptions."""
    print("\n--- Phase 3: Description Dedup ---")

    rows = await conn.fetch(
        "SELECT id, description FROM entities WHERE description LIKE '%\n%'"
    )
    print(f"  Entities with multiline descriptions: {len(rows)}")

    update_count = 0
    for r in rows:
        cleaned = dedup_description(r["description"])
        if cleaned != r["description"]:
            update_count += 1
            if apply:
                await conn.execute(
                    "UPDATE entities SET description = $1 WHERE id = $2",
                    cleaned, r["id"],
                )

    print(f"  Descriptions cleaned: {update_count}")


# ---------------------------------------------------------------------------
# Phase 4: Orphan cleanup
# ---------------------------------------------------------------------------

async def phase_orphan_cleanup(conn: asyncpg.Connection, *, apply: bool) -> None:
    """Delete orphaned entities with low mention count."""
    print("\n--- Phase 4: Orphan Cleanup ---")

    orphans = await conn.fetch("""
        SELECT e.id, e.name, e.mention_count
        FROM entities e
        LEFT JOIN relationships r ON r.source_id = e.id OR r.target_id = e.id
        WHERE r.id IS NULL
        ORDER BY e.mention_count DESC
    """)
    print(f"  Total orphans: {len(orphans)}")

    to_delete = [o for o in orphans if o["mention_count"] < 3]
    to_keep = [o for o in orphans if o["mention_count"] >= 3]

    print(f"  Will delete (mention_count < 3): {len(to_delete)}")
    if to_keep:
        print(f"  Preserving (mention_count >= 3): {len(to_keep)}")
        for o in to_keep[:10]:
            print(f"    '{o['name']}' (mentions={o['mention_count']})")

    if apply and to_delete:
        ids = [o["id"] for o in to_delete]
        await conn.execute("DELETE FROM entities WHERE id = ANY($1::uuid[])", ids)


# ---------------------------------------------------------------------------
# Phase 5: VACUUM ANALYZE
# ---------------------------------------------------------------------------

async def phase_vacuum(conn: asyncpg.Connection, *, apply: bool) -> None:
    """Run VACUUM ANALYZE on relevant tables."""
    print("\n--- Phase 5: VACUUM ANALYZE ---")

    tables = ["entities", "relationships", "communities", "community_members", "conversation_messages"]

    if not apply:
        print("  Would VACUUM ANALYZE: " + ", ".join(tables))
        return

    # VACUUM cannot run inside a transaction
    for table in tables:
        print(f"  VACUUM ANALYZE {table}...")
        await conn.execute(f"VACUUM ANALYZE {table}")  # noqa: S608

    print("  Done.")


# ---------------------------------------------------------------------------
# Phase 6: Community rebuild
# ---------------------------------------------------------------------------

async def phase_community_rebuild(conn: asyncpg.Connection, *, apply: bool) -> None:
    """Rebuild communities using existing CommunityDetector."""
    print("\n--- Phase 6: Community Rebuild ---")

    if not apply:
        print("  Would rebuild communities (requires API keys).")
        return

    # Import here to avoid requiring API keys for dry-run/check
    from prot.community import CommunityDetector  # noqa: PLC0415
    from prot.embeddings import AsyncVoyageEmbedder  # noqa: PLC0415
    from prot.graphrag import GraphRAGStore  # noqa: PLC0415

    # Create a temporary pool for GraphRAGStore
    from prot.config import settings  # noqa: PLC0415
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=3)
    try:
        store = GraphRAGStore(pool)
        embedder = AsyncVoyageEmbedder()
        detector = CommunityDetector(store, embedder)
        try:
            count = await detector.rebuild()
            print(f"  Rebuilt {count} communities.")
        finally:
            await detector.close()
            await embedder.close()
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

PHASES = {
    "entity-dedup": phase_entity_dedup,
    "relation-norm": phase_relation_norm,
    "desc-dedup": phase_desc_dedup,
    "orphan-cleanup": phase_orphan_cleanup,
    "vacuum": phase_vacuum,
    "community-rebuild": phase_community_rebuild,
}


async def run_full(
    conn: asyncpg.Connection,
    *,
    apply: bool,
    skip_rebuild: bool = False,
    skip_vacuum: bool = False,
) -> None:
    """Run all GC phases in order."""
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n{'='*60}")
    print(f"GraphRAG GC — Full Run ({mode})")
    print(f"{'='*60}")

    await phase_entity_dedup(conn, apply=apply)
    await phase_relation_norm(conn, apply=apply)
    await phase_desc_dedup(conn, apply=apply)
    await phase_orphan_cleanup(conn, apply=apply)

    if skip_vacuum:
        print("\n--- Phase 5: VACUUM ANALYZE (SKIPPED) ---")
    else:
        await phase_vacuum(conn, apply=apply)

    if skip_rebuild:
        print("\n--- Phase 6: Community Rebuild (SKIPPED) ---")
    else:
        await phase_community_rebuild(conn, apply=apply)

    print(f"\n{'='*60}")
    if apply:
        print("WARNING: Restart prot to refresh the entity cache.")
    print("Done.")


async def run_phase(conn: asyncpg.Connection, phase_name: str, *, apply: bool) -> None:
    """Run a single GC phase."""
    fn = PHASES[phase_name]
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\nGraphRAG GC — {phase_name} ({mode})")
    await fn(conn, apply=apply)
    if apply:
        print("\nWARNING: Restart prot to refresh the entity cache.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG database GC and dedup")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Read-only diagnostic report")

    full_p = sub.add_parser("full", help="Run all GC phases")
    full_p.add_argument("--apply", action="store_true", help="Actually apply changes (default: dry-run)")
    full_p.add_argument("--skip-rebuild", action="store_true", help="Skip community rebuild")
    full_p.add_argument("--skip-vacuum", action="store_true", help="Skip VACUUM ANALYZE")

    phase_p = sub.add_parser("phase", help="Run a single GC phase")
    phase_p.add_argument("name", choices=list(PHASES.keys()), help="Phase to run")
    phase_p.add_argument("--apply", action="store_true", help="Actually apply changes (default: dry-run)")

    args = parser.parse_args()

    conn = await _connect()
    try:
        if args.command == "check":
            await run_check(conn)
        elif args.command == "full":
            await run_full(
                conn,
                apply=args.apply,
                skip_rebuild=args.skip_rebuild,
                skip_vacuum=args.skip_vacuum,
            )
        elif args.command == "phase":
            await run_phase(conn, args.name, apply=args.apply)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
