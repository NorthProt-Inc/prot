# Token-Budget Context Management

**Date**: 2026-02-19
**Status**: Approved

## Problem

The current sliding window (`context_max_turns=10`) has three issues:

1. **Long conversations lose early context** — important turns from >10 turns ago are discarded
2. **Short conversations waste budget** — brief sessions still send 20 messages of budget capacity
3. **Tool calls eat conversation turns** — tool_use/tool_result pairs consume slots meant for real dialogue

## Solution: Token-Budget Sliding Window

Replace fixed turn count with a **token budget**. Fill from newest messages backwards until the budget is exhausted.

### Key Constraints

- **Zero additional latency in tool loop** — use `usage.input_tokens` from previous response instead of re-calling count_tokens
- **ContextManager stays pure** — no API client dependency; token budgeting lives in a separate `TokenBudgetTrimmer`
- **Exact token counting** — Anthropic `count_tokens()` API on first call per utterance

## Architecture

```
pipeline._process_response()
  |-- system_blocks = ctx.build_system_blocks()
  |-- tools = ctx.build_tools(...)
  |-- trimmer = TokenBudgetTrimmer(llm, model, system, tools, budget)
  |
  |-- for iteration in tool_loop:
  |     |-- messages = ctx.get_recent_messages()   # sync, returns all (orphan-corrected)
  |     |-- messages = await trimmer.fit(messages)  # trim to budget
  |     |-- async for chunk in llm.stream_response(...)
  |     +-- trimmer.update_overhead(llm.last_usage)  # free token info from response
```

### TokenBudgetTrimmer (new: src/prot/trimmer.py)

```python
class TokenBudgetTrimmer:
    """Trim messages to fit within a token budget.

    First call: uses count_tokens() API for exact count (~50-100ms).
    Subsequent calls (tool loop): uses previous response's usage.input_tokens
    to avoid additional API calls.
    """

    def __init__(self, client, model, system, tools, budget): ...

    async def fit(self, messages: list[dict]) -> list[dict]:
        """Return messages trimmed to fit within token budget.

        Strategy:
        1. If overhead is unknown (first call), count via API
        2. If over budget, remove oldest complete exchanges first
        3. Truncate long tool_results as last resort
        4. Ensure window starts at valid user message (not orphaned tool_result)
        """
        ...

    def update_overhead(self, usage) -> None:
        """Update known token overhead from response usage.input_tokens."""
        ...
```

**First call flow**:
1. Call `count_tokens(model, system, tools, messages, thinking={"type":"adaptive"})` — exact count
2. If over budget, remove oldest complete exchange (user+assistant pair) from front
3. Re-count or estimate delta until within budget
4. Truncate long tool_results (>context_tool_result_max_chars) as last resort

**Tool loop flow** (2nd+ iteration):
1. Previous `usage.input_tokens` is known from response
2. Estimate new tool_result tokens via `len(content) // 4`
3. If estimated total > budget, trim from front
4. No count_tokens API call needed

### LLMClient Changes (src/prot/llm.py)

- Add `last_usage` property (stores `final.usage` from `get_final_message()`)
- Add `count_tokens(system, tools, messages) -> int` method
- `count_tokens` includes `thinking={"type": "adaptive"}` to match `stream_response` params

### Config Changes (src/prot/config.py)

- Remove: `context_max_turns: int = 10`
- Add: `context_token_budget: int = 30000` (messages-only budget, conservative vs 200k limit)
- Add: `context_tool_result_max_chars: int = 2000` (truncate tool_results beyond this)

### ContextManager Changes (src/prot/context.py)

- `get_recent_messages()` simplified: remove `max_turns` parameter, return all messages with orphan correction only
- No async change, no API dependency

## Edge Cases

| Case | Handling |
|------|----------|
| tool_use/tool_result pair split | Remove only complete exchanges (user+assistant pairs) |
| web_search_tool_result | Encrypted content — exclude from truncation |
| Empty messages (first utterance) | No trimming needed, return as-is |
| Budget smaller than 1 message | Always include at least the last user message |
| count_tokens API failure | Fallback to `len // 4` heuristic |
| Orphaned tool_result at window start | Skip forward to first real user message |

## Test Impact

- `test_context.py`: No changes (ContextManager unchanged)
- `test_pipeline.py`: Add trimmer mock (minor)
- `test_trimmer.py`: New file — unit tests for trim logic with mocked count_tokens

## Files Changed

| File | Change |
|------|--------|
| `src/prot/trimmer.py` | **New** — TokenBudgetTrimmer class |
| `src/prot/llm.py` | Add `last_usage` property, `count_tokens()` method |
| `src/prot/config.py` | Replace `context_max_turns` with `context_token_budget` + `context_tool_result_max_chars` |
| `src/prot/context.py` | Simplify `get_recent_messages()` — remove max_turns param |
| `src/prot/pipeline.py` | Integrate trimmer in `_process_response()` |
| `tests/test_trimmer.py` | **New** — trimmer unit tests |
| `tests/test_pipeline.py` | Update mocks for trimmer |
