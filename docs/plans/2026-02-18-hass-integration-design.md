# HASS Integration Redesign — 2-Tool Split with Auto-Discovery

**Date:** 2026-02-18
**Status:** Approved

## Problem Statement

The current Home Assistant integration has several issues:
1. `entity_id` is a free-form string — LLM hallucinates invalid IDs
2. `call_service` uses `entity_id.split(".", 1)` as domain/service — semantically wrong
3. No `color_temp_kelvin` support — only RGB, causing color mismatch for WiZ lights
4. `tools: Input should be a valid list` API error from two code paths:
   - `pipeline.py:230` sets `iter_tools = None` on final iteration
   - `llm.py:44` converts empty list to `None`
5. No auto-discovery — requires manual entity management

## Requirements

- Fully automatic entity discovery from HASS API (no YAML, no manual config)
- Prevent LLM hallucination via enum constraints on entity_id
- Full service_data: `color_temp_kelvin` (2200-6500K) AND `rgb_color` for lights
- On/off only for fan
- Minimal maintenance — startup-only discovery, app restart for new devices
- 2-tool split: `hass_control` (state changes) + `hass_query` (state reads)

## Architecture

### New Module: `src/prot/hass.py`

Single file containing:

```
HassRegistry
├── __init__(url, token)           # Creates httpx.AsyncClient
├── async discover()               # GET /api/states + /api/services, startup-only
├── build_tool_schemas() -> list   # Returns hass_control + hass_query tool defs
├── async execute_control(...)     # Executes turn_on/off/toggle
├── async execute_query(...)       # Executes get_state/list_entities
└── async close()                  # Closes httpx client

parse_color(input: str) -> list[int] | None   # Standalone pure function
```

### Auto-Discovery

- Called once in `Pipeline.startup()` (no periodic refresh)
- `GET /api/states` → enumerate all entities
- `GET /api/services` → enumerate available services per domain
- **Domain allowlist:** `light`, `fan`, `weather`, `sensor`, `switch`, `climate`
- Entities outside allowlist are silently excluded
- HASS connection failure → graceful degradation (no HASS tools, app still works)

### Tool Schemas

**hass_control** — state changes:
```json
{
  "name": "hass_control",
  "description": "Control Home Assistant device.\nAvailable: light.wiz_1 (WiZ RGBW), ..., fan.vital_100s (Vital 100S)\ncolor and color_temp_kelvin are mutually exclusive; color_temp_kelvin takes priority.",
  "input_schema": {
    "type": "object",
    "properties": {
      "entity_id": {"type": "string", "enum": ["<populated from discover>"]},
      "action": {"type": "string", "enum": ["turn_on", "turn_off", "toggle"]},
      "brightness": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Brightness percentage (lights only)"},
      "color": {"type": "string", "description": "Color name (red, 빨강, warm, #FF0000) — lights only"},
      "color_temp_kelvin": {"type": "integer", "minimum": 2200, "maximum": 6500, "description": "Color temperature in Kelvin — lights only"}
    },
    "required": ["entity_id", "action"]
  }
}
```

**hass_query** — state reads:
```json
{
  "name": "hass_query",
  "description": "Query Home Assistant entity states.",
  "input_schema": {
    "type": "object",
    "properties": {
      "entity_id": {"type": "string", "enum": ["<populated from discover>"]},
      "query_type": {"type": "string", "enum": ["get_state", "list_entities"]}
    },
    "required": ["query_type"]
  }
}
```

- Last tool in `build_tools()` gets `cache_control: {"type": "ephemeral"}`
- `entity_id` is optional for `list_entities`, validated as required for `get_state`

### Execution Layer

**hass_control flow:**
```
LLM: hass_control(entity_id="light.wiz_1", action="turn_on", color_temp_kelvin=4000)
  → HassRegistry.execute_control()
    → domain = "light" (from entity_id.split(".")[0])
    → service_data = {"entity_id": "light.wiz_1", "color_temp_kelvin": 4000}
    → POST /api/services/light/turn_on  body=service_data
    → return {"success": True, "message": "light.wiz_1 turned on (4000K)"}
```

**Domain-aware parameter filtering:**
- `brightness`, `color`, `color_temp_kelvin` only included when domain is `light`
- Other domains: only `entity_id` in service_data

**Brightness conversion:**
- Schema accepts 0-100 (percentage, intuitive for LLM)
- Execution layer uses `brightness_pct` field in HASS service_data (native 0-100 support)
- Alternatively: convert to 0-255 via `int(brightness * 255 / 100)` for `brightness` field

**Color handling priority:**
1. `color_temp_kelvin` present → `{"color_temp_kelvin": N}` (direct passthrough)
2. `color` present → `parse_color()` → `{"rgb_color": [R, G, B]}`
3. Both present → `color_temp_kelvin` wins (mutually exclusive on WiZ lights)

**parse_color()** supports:
- Named colors: English (red, blue, warm, cool) + Korean (빨강, 파랑, 초록, 노랑, 분홍, 보라, 주황, 하양, 흰색)
- Hex: `#FF0000` or `FF0000`
- HSL: `hsl(120, 100, 50)` or `120, 100, 50`
- RGB: `rgb(255, 0, 0)`

**Return type:**
- Success: `{"success": True, "message": "light.wiz_1 turned on (warm, 80%)"}`
- Error: `{"error": "Invalid entity_id: light.nonexistent"}`

### Integration Points

**`context.py` — `build_tools(hass_registry=None)`:**
- When `hass_registry` is None → return `[web_search]`
- When present → return `[web_search, hass_control_schema, hass_query_schema]`
- Last tool always gets `cache_control: {"type": "ephemeral"}`

**`pipeline.py` — tool routing in `_process_response()`:**
- Pipeline routes tool calls directly (not through LLMClient):
  ```python
  if block.name in ("hass_control", "hass_query"):
      result = await self._hass_registry.execute(block.name, block.input)
  else:
      result = await self._llm.execute_tool(block.name, block.input)
  ```
- `startup()` initializes HassRegistry before other optional components
- `shutdown()` calls `registry.close()`

**`llm.py` — cleanup:**
- Remove `_execute_hass()` method
- Remove `self._hass_client` (httpx client moves to HassRegistry)
- Remove `execute_tool()` for HASS (keep for future non-HASS tools if needed)
- Fix line 44: `tools=tools` (always pass list, never convert to None)

## Bug Fixes

### Fix 1: `pipeline.py:230` — iter_tools = None
```python
# BEFORE (buggy)
iter_tools = tools if iteration < self._MAX_TOOL_ITERATIONS - 1 else None

# AFTER — remove iter_tools entirely, always pass tools
async for chunk in self._llm.stream_response(system_blocks, tools, messages):
```

### Fix 2: `llm.py:44` — tools falsy conversion
```python
# BEFORE (buggy — empty list becomes None)
tools=tools if tools else None,

# AFTER — always pass what caller provides
tools=tools,
```

## Breaking Tests (10 total)

| Test File | Test Name | Action |
|-----------|-----------|--------|
| test_llm.py | test_execute_tool_unknown_returns_error | Update tool routing |
| test_llm.py | test_execute_hass_invalid_entity_id_returns_error | Move to test_hass.py |
| test_llm.py | test_execute_hass_missing_entity_id_returns_error | Move to test_hass.py |
| test_llm.py | test_execute_hass_get_state_success | Move to test_hass.py |
| test_llm.py | test_execute_hass_http_error_returns_error | Move to test_hass.py |
| test_llm.py | test_close_closes_hass_client | Remove (httpx moves to HassRegistry) |
| test_context.py | test_build_tools_returns_list | Update for new signature |
| test_pipeline.py | test_tool_use_executes_and_loops | Update tool names + routing |
| test_pipeline.py | test_tool_use_error_is_reported | Update tool names + routing |
| test_pipeline.py | test_bails_out_when_interrupted_during_tool_exec | Update tool names |

New test file: `tests/test_hass.py` — HassRegistry unit tests, parse_color tests.

Test helper `_make_pipeline()` in test_pipeline.py must add `_hass_registry = None`.

## Known Limitations (v1)

- **No "all lights" batch control** — LLM calls hass_control once per light (acceptable for 6 lights)
- **Startup-only discovery** — new devices require app restart
- **No circuit breaker / retry** — single attempt per API call (HASS is on local network)

## Files Changed

| File | Type | Description |
|------|------|-------------|
| `src/prot/hass.py` | NEW | HassRegistry + parse_color |
| `src/prot/context.py` | MODIFY | build_tools() accepts hass_registry param |
| `src/prot/pipeline.py` | MODIFY | startup() init, tool routing, bug fix |
| `src/prot/llm.py` | MODIFY | Remove HASS code, fix tools=None bug |
| `tests/test_hass.py` | NEW | HassRegistry + parse_color tests |
| `tests/test_llm.py` | MODIFY | Remove HASS tests, update execute_tool |
| `tests/test_context.py` | MODIFY | Update build_tools test |
| `tests/test_pipeline.py` | MODIFY | Update tool names + routing |
