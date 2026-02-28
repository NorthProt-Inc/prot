# Server-side Context Compaction + Context Editing

Migrate from client-side `TokenBudgetTrimmer` to Anthropic's server-side
context management: compaction, tool result clearing, and thinking block
clearing. All three run server-side before the prompt reaches Claude,
eliminating custom trimming logic entirely.

## Motivation

- `TokenBudgetTrimmer` is a hand-rolled heuristic (char/4 estimate, oldest-exchange
  removal). Server-side compaction produces an actual LLM summary.
- Tool result clearing handles `web_search` and `hass_request` bloat automatically.
- Thinking block clearing manages adaptive thinking overhead without manual stripping.
- Sonnet 4.6 now supports server-side compaction (`compact-2026-01-12` beta).

## Architecture Change

### Before

```
messages -> TokenBudgetTrimmer.fit() -> trimmed messages -> LLM (GA API)
             |-- tool_result truncation (max_chars)
             |-- oldest exchange removal
             |-- count_tokens API call
```

### After

```
messages -> LLM (Beta API) -> server automatically applies:
             1. thinking block clearing (always, keep 2 turns)
             2. tool result clearing   (>30K tokens)
             3. compaction summary     (>50K tokens)
```

## Beta Headers

Two beta headers required simultaneously:

- `compact-2026-01-12` — server-side compaction
- `context-management-2025-06-27` — tool result clearing + thinking block clearing

## Configuration

New settings in `config.py` (replacing removed settings):

| Setting | Default | Description |
|---------|---------|-------------|
| `compaction_trigger` | 50000 | Token count that triggers compaction (min 50K) |
| `tool_clear_trigger` | 30000 | Token count that triggers tool result clearing |
| `tool_clear_keep` | 3 | Recent tool use/result pairs to preserve |
| `thinking_keep_turns` | 2 | Recent assistant turns with thinking to preserve |

Removed settings:

- `context_token_budget` (30000) — replaced by `compaction_trigger`
- `context_tool_result_max_chars` (2000) — replaced by server-side `clear_tool_uses`

## API Changes

### LLM Client (`llm.py`)

Switch from GA to Beta API:

```python
async with self._client.beta.messages.stream(
    model=settings.claude_model,
    max_tokens=settings.claude_max_tokens,
    thinking={"type": "adaptive"},
    output_config={"effort": settings.claude_effort},
    system=system_blocks,
    tools=tools,
    messages=messages,
    betas=["compact-2026-01-12", "context-management-2025-06-27"],
    context_management={
        "edits": [
            {
                "type": "clear_thinking_20251015",
                "keep": {"type": "thinking_turns", "value": 2},
            },
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
            },
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 50000},
            },
        ],
    },
) as stream:
```

Ordering matters: thinking clearing first, then tool clearing, then compaction.

`count_tokens` also migrates to `client.beta.messages.count_tokens` with
matching `betas` and `context_management` params.

### Compaction Block Handling

When compaction triggers, the response includes a `compaction` content block:

```json
{
  "content": [
    {"type": "compaction", "content": "Summary of conversation..."},
    {"type": "text", "text": "Based on our conversation..."}
  ]
}
```

The full response content (including the compaction block) is stored in
`ContextManager._messages` as-is. On the next request, the API automatically
drops all messages before the compaction block.

### Usage Tracking

With compaction enabled, `usage.iterations` array replaces simple
`usage.input_tokens` for accurate billing:

```json
{
  "usage": {
    "input_tokens": 45000,
    "output_tokens": 1234,
    "iterations": [
      {"type": "compaction", "input_tokens": 180000, "output_tokens": 3500},
      {"type": "message", "input_tokens": 23000, "output_tokens": 1000}
    ]
  }
}
```

Top-level `input_tokens`/`output_tokens` exclude compaction iterations.

## Files Changed

| File | Change |
|------|--------|
| `trimmer.py` | **Delete** |
| `test_trimmer.py` | **Delete** |
| `config.py` | Remove `context_token_budget`, `context_tool_result_max_chars`. Add `compaction_trigger`, `tool_clear_trigger`, `tool_clear_keep`, `thinking_keep_turns` |
| `llm.py` | GA -> Beta API. Add `betas` + `context_management` params. Update `count_tokens` to beta. Remove `_last_usage` overhead tracking if unused. |
| `pipeline.py` | Remove `TokenBudgetTrimmer` import, instantiation, `trimmer.fit()`, `trimmer.update_overhead()`. Keep `ContextManager.get_recent_messages()` for orphan boundary fix. |
| `context.py` | No change (compaction blocks stored as normal message content) |

## Prompt Caching Interaction

System prompt already uses `cache_control: ephemeral` on blocks 1-2. When
compaction occurs, the system prompt cache remains valid. Only the compaction
summary gets written as a new cache entry. This is the optimal pattern per
Anthropic docs.

## Future Work (deferred)

- Compaction results -> RAG/memory integration
- Restart context loading from previous compaction summary
- Custom `instructions` for domain-specific summarization
