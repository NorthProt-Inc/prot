# HA Agent Delegation Design

Date: 2026-02-27

## Problem

prot's current HA integration routes all smart home actions through Claude's tool loop:
- `hass_control` + `hass_query` tools with complex enum-based schemas (~500 tokens)
- Every action requires Claude to reason about HA entities, domains, and parameters
- 2-10 second latency per action (Claude reasoning + REST API + optional re-loop)
- Token waste on HA-specific logic that HA's built-in conversation agent already handles
- Color parsing, entity management, and domain logic maintained in prot

## Solution

Replace `hass_control` + `hass_query` with a single `hass_request` tool that delegates to HA's conversation agent (Gemini Flash) via `/api/conversation/process`.

### Approach: Tool Replacement + Schema Minimization

- Replace two complex tools with one simple tool (`hass_request`)
- Claude interprets user intent and formulates a clear command
- HA's conversation agent (Gemini Flash) handles entity resolution and service calls
- Claude re-phrases HA's response in Axel persona
- Each request is stateless (no conversation_id persistence)

## New Tool Schema

```python
{
    "name": "hass_request",
    "description": "Send a natural language command to the Home Assistant agent. "
                   "Use for any smart home task: device control, state queries, "
                   "automation triggers, scene activation, etc.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Natural language command for Home Assistant. "
                               "Be specific with device names and values. "
                               "Example: '조명 밝기 40%, 색온도 2700K로 변경'"
            }
        },
        "required": ["command"]
    }
}
```

Token savings: ~500 tokens → ~50 tokens per LLM call.

## Architecture

### New: HassAgent class (~40 lines)

```python
class HassAgent:
    def __init__(self, url: str, token: str):
        self._url = url.rstrip("/")
        self._token = token
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))

    async def request(self, command: str) -> str:
        """Send command to HA conversation agent, return response text."""
        resp = await self._client.post(
            f"{self._url}/api/conversation/process",
            headers={"Authorization": f"Bearer {self._token}"},
            json={"text": command, "language": "ko"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["response"]["speech"]["plain"]["speech"]

    def build_tool(self) -> dict:
        """Return single tool schema for Claude."""
        return { ... }
```

### Removed

- `HassClient` class (get_state, call_service, list_entities)
- `HassRegistry` class (entity cache, enum generation, execute dispatch)
- `_parse_color()` and all color-related logic
- `_HASS_ENTITY_SCHEMA`, `_HASS_CONTROL_SCHEMA`, `_HASS_QUERY_SCHEMA`
- Startup entity scan HTTP call

## Data Flow

```
User: "불 좀 따뜻하게 해줘"
  ↓
[Claude] interprets → tool_use: hass_request
  command: "조명 색온도 3000K로 변경"
  ↓
[prot pipeline] → POST /api/conversation/process
  {"text": "조명 색온도 3000K로 변경", "language": "ko"}
  ↓
[HA Gemini Flash] → resolves entities → calls light.turn_on service
  ↓
[HA response] → "조명을 3000K로 변경했습니다"
  ↓
[Claude] re-phrases in Axel persona → TTS → Speaker
```

## File Changes

| File | Change |
|------|--------|
| `src/prot/hass.py` | Full rewrite (~40 lines, HassAgent class) |
| `src/prot/pipeline.py` | HassRegistry → HassAgent replacement |
| `src/prot/context.py` | build_tools signature change |
| `src/prot/config.py` | No change (hass_url, hass_token retained) |
| `tests/test_hass.py` | Full rewrite |
| `tests/test_pipeline.py` | Mock target change |
| `tests/test_context.py` | build_tools call update |

## Error Handling

1. **HA server down** (ConnectionError) → tool_result: "Home Assistant에 연결할 수 없습니다"
2. **Auth failure** (401) → tool_result: "Home Assistant 인증 실패" + log warning
3. **Gemini agent error** → error message in tool_result → Claude explains to user
4. **Timeout** (15s) → tool_result: "Home Assistant 응답 시간 초과"

Graceful degradation unchanged: no hass_token → tool not exposed to Claude.

## Environment Context

- Single zone: NorthProt
- 6x Wiz RGBW Tunable lights
- 1x Vital 100S air purifier
- HA conversation agent: Gemini 3 Flash (primary)
- Location info unnecessary in commands

## Testing

- Unit: HassAgent.request() with httpx mock (success, connection error, auth failure, timeout)
- Unit: build_tool() schema validation
- Integration: Real HA conversation/process call with light control command
