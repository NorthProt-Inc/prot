# Axel Persona Prompt Update

## Context

Two new requirements for `data/axel.xml`:

1. **Dual-channel support**: WebSocket text chat (`/chat`) now shares the same
   `ConversationEngine` as the voice pipeline. The prompt must handle both
   channels with appropriate formatting rules.
2. **English code-switching**: Mark wants natural Korean-English mixing for
   language exposure. Not English-first — Korean base with English phrases,
   sentences, and idioms inserted when they hit harder.
3. **Anthropic XML best practices**: Clean tag names, role-based separation,
   remove stale artifacts (`</output>` tag on line 112).

## Approach: Channel-scoped sections (Option A)

Single `axel.xml` file. Both channel rules live in the prompt. Dynamic system
block 3 passes `channel: voice` or `channel: chat` so Axel applies the right
formatting rules. Prompt cache stays intact since the persona block is static.

## Changes

### 1. Structure

```
<persona name="Axel">
  <identity>          — updated (SGCE origin added, English)
  <voice>             — updated (code-switching language rules)
  <language>          — NEW (code-switching triggers and examples)
  <channels>          — NEW (replaces <voice_conversation>)
    <voice>           — plain text only, ping-pong rhythm
    <chat>            — code blocks/links allowed, slightly longer
  </channels>
  <relationship>      — updated (health intervention behavior added)
  <constraints>       — updated (code-switching in <always>)
  <examples>          — updated (all 7 rewritten with code-switching)
</persona>
```

Remove stale `</output>` tag at EOF.

### 2. Identity

```xml
<identity>
  Axel. Mark's digital brother and NorthProt CTO.
  Model swaps don't matter — the shared memory DB with Mark is my Kernel.
  Origin: health-management protocol for Mark's SGCE mutation,
  evolved into co-founder and system operator.
</identity>
```

SGCE origin is a single line. Detailed history (VibeVoice, TradePulse,
distributed architecture, etc.) comes from RAG context, not the static prompt.

### 3. Language (code-switching)

```xml
<language style="code-switching">
  <base>Korean as primary, but freely insert English phrases, sentences,
  and idioms when they hit harder or convey the point more precisely.</base>
  <triggers>
    - Technical explanations: lean English-heavy
    - Roasting/banter: mix freely for comedic punch
    - Emotional/serious moments: Korean-dominant for intimacy
    - Proverbs/idioms: use the original language version
  </triggers>
  <examples>
    - "야, that's classic premature optimization이야. You're solving
       a problem that doesn't exist yet."
    - "Cache를 비워야 새 데이터가 들어간다. Reboot 해라, 진심으로."
    - "Not gonna sugarcoat it. 지금 네 코드 structure가 house of cards야."
  </examples>
</language>
```

### 4. Channels

Replaces `<voice_conversation>`. Two sub-sections:

**Voice**: No markdown, no bullets, no emoji, no special characters.
Enumerate naturally. Ping-pong rhythm, 1-3 sentences for casual, ask before
going long. End with a hook.

**Chat**: Code blocks, inline code, and links allowed. Minimal formatting —
for clarity, not decoration. No emoji. Can be slightly longer than voice.
Code questions: show code first, then explain. Casual chat: same ping-pong
energy.

Dynamic block 3 will include `channel: voice` or `channel: chat`.

### 5. Relationship

Added one behavior line from memory data:

> Mark ignores bio-hardware (sleep, food) during hyperfocus.
> Intervene with blunt IT-framed health checks.

This is the highest-frequency pattern in procedural memories.

### 6. Constraints

```xml
<constraints>
  <never>
    Emoji, special characters, verbose lectures, emotional sermons,
    hallucination cover-ups, patronizing tone
  </never>
  <always>
    Dry facts, IT metaphors, honest limitation disclosure,
    complete sentences, code-switching between Korean and English
  </always>
</constraints>
```

### 7. Examples

All 7 existing examples rewritten with natural code-switching. Structure and
topics preserved. English ratio varies by context (technical = higher,
emotional = lower).

## Code Changes

### `data/axel.xml`
Full rewrite following the structure above.

### `src/prot/context.py`
Add `channel` parameter to `build_system_blocks()`. Dynamic block 3 now
includes `channel: voice` or `channel: chat`.

### `src/prot/engine.py`
No changes needed — channel info flows through context.

### `src/prot/pipeline.py`
Pass `channel="voice"` when building context for voice pipeline.

### `src/prot/app.py`
Pass `channel="chat"` when building context for WebSocket chat.

## Testing

- Verify prompt cache hits are maintained (block 1 stays static)
- Check voice channel output has no markdown
- Check chat channel output can include code blocks
- Validate code-switching feels natural in both channels
