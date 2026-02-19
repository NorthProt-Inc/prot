import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx
from prot.hass import parse_color, HassRegistry


class TestParseColor:
    def test_named_english_red(self):
        assert parse_color("red") == [255, 0, 0]

    def test_named_english_blue(self):
        assert parse_color("blue") == [0, 0, 255]

    def test_named_english_warm(self):
        assert parse_color("warm") == [255, 180, 107]

    def test_named_english_cool(self):
        assert parse_color("cool") == [166, 209, 255]

    def test_named_korean_빨강(self):
        assert parse_color("빨강") == [255, 0, 0]

    def test_named_korean_파랑(self):
        assert parse_color("파랑") == [0, 0, 255]

    def test_named_korean_초록(self):
        assert parse_color("초록") == [0, 128, 0]

    def test_named_korean_노랑(self):
        assert parse_color("노랑") == [255, 255, 0]

    def test_named_korean_분홍(self):
        assert parse_color("분홍") == [255, 192, 203]

    def test_named_korean_보라(self):
        assert parse_color("보라") == [128, 0, 128]

    def test_named_korean_주황(self):
        assert parse_color("주황") == [255, 165, 0]

    def test_named_korean_하양(self):
        assert parse_color("하양") == [255, 255, 255]

    def test_named_korean_흰색(self):
        assert parse_color("흰색") == [255, 255, 255]

    def test_hex_with_hash(self):
        assert parse_color("#FF0000") == [255, 0, 0]

    def test_hex_without_hash(self):
        assert parse_color("00FF00") == [0, 255, 0]

    def test_hex_lowercase(self):
        assert parse_color("#ff8800") == [255, 136, 0]

    def test_rgb_format(self):
        assert parse_color("rgb(255, 0, 0)") == [255, 0, 0]

    def test_hsl_format(self):
        result = parse_color("hsl(120, 100, 50)")
        assert result == [0, 255, 0]

    def test_hsl_bare_format(self):
        result = parse_color("120, 100, 50")
        assert result == [0, 255, 0]

    def test_unknown_returns_none(self):
        assert parse_color("xyzzy") is None

    def test_empty_returns_none(self):
        assert parse_color("") is None

    def test_case_insensitive(self):
        assert parse_color("RED") == [255, 0, 0]
        assert parse_color("Blue") == [0, 0, 255]


class TestHassRegistryDiscover:
    async def test_discover_filters_by_domain_allowlist(self):
        """Only entities from allowed domains (light, fan, switch, etc.) are kept."""
        mock_states = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ RGBW"}},
            {"entity_id": "fan.vital_100s", "attributes": {"friendly_name": "Vital 100S"}},
            {"entity_id": "camera.front", "attributes": {"friendly_name": "Front Camera"}},
        ]
        registry = HassRegistry("http://hass:8123", "token")
        registry._client = AsyncMock()
        registry._client.get = AsyncMock(return_value=MagicMock(
            status_code=200, json=MagicMock(return_value=mock_states),
        ))
        await registry.discover()

        ids = [e["entity_id"] for e in registry._entities]
        assert "light.wiz_1" in ids
        assert "fan.vital_100s" in ids
        assert "camera.front" not in ids

    async def test_discover_failure_sets_empty(self):
        """HASS connection failure -> graceful degradation."""
        registry = HassRegistry("http://hass:8123", "token")
        registry._client = AsyncMock()
        registry._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        await registry.discover()

        assert registry._entities == []

    async def test_discover_non_200_sets_empty(self):
        """Non-200 response -> graceful degradation."""
        registry = HassRegistry("http://hass:8123", "token")
        registry._client = AsyncMock()
        registry._client.get = AsyncMock(return_value=MagicMock(status_code=401))
        await registry.discover()

        assert registry._entities == []


class TestHassRegistryBuildToolSchemas:
    async def test_build_returns_two_tools_when_entities_exist(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ RGBW"}},
        ]
        tools = registry.build_tool_schemas()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"hass_control", "hass_query"}

    async def test_build_returns_empty_when_no_entities(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = []
        tools = registry.build_tool_schemas()
        assert tools == []

    async def test_entity_ids_in_enum(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ RGBW"}},
            {"entity_id": "fan.vital_100s", "attributes": {"friendly_name": "Vital 100S"}},
        ]
        tools = registry.build_tool_schemas()
        control = next(t for t in tools if t["name"] == "hass_control")
        entity_enum = control["input_schema"]["properties"]["entity_id"]["enum"]
        assert "light.wiz_1" in entity_enum
        assert "fan.vital_100s" in entity_enum

    async def test_last_tool_has_cache_control(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ RGBW"}},
        ]
        tools = registry.build_tool_schemas()
        assert "cache_control" in tools[-1]
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}

    async def test_hass_control_has_brightness_and_color_fields(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        tools = registry.build_tool_schemas()
        control = next(t for t in tools if t["name"] == "hass_control")
        props = control["input_schema"]["properties"]
        assert "brightness" in props
        assert "color" in props
        assert "color_temp_kelvin" in props

    async def test_hass_query_has_query_type_enum(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        tools = registry.build_tool_schemas()
        query = next(t for t in tools if t["name"] == "hass_query")
        query_type = query["input_schema"]["properties"]["query_type"]
        assert query_type["enum"] == ["get_state", "list_entities"]

    async def test_description_contains_entity_names(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ RGBW"}},
        ]
        tools = registry.build_tool_schemas()
        control = next(t for t in tools if t["name"] == "hass_control")
        assert "light.wiz_1" in control["description"]
        assert "WiZ RGBW" in control["description"]


class TestHassRegistryExecuteControl:
    async def test_turn_on_light_with_kelvin(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        registry._client = AsyncMock()
        registry._client.post = AsyncMock(return_value=MagicMock(status_code=200))

        result = await registry.execute_control({
            "entity_id": "light.wiz_1",
            "action": "turn_on",
            "color_temp_kelvin": 4000,
        })
        assert result["success"] is True
        call_args = registry._client.post.call_args
        assert "/api/services/light/turn_on" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["entity_id"] == "light.wiz_1"
        assert body["color_temp_kelvin"] == 4000

    async def test_turn_on_light_with_color_name(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        registry._client = AsyncMock()
        registry._client.post = AsyncMock(return_value=MagicMock(status_code=200))

        result = await registry.execute_control({
            "entity_id": "light.wiz_1",
            "action": "turn_on",
            "color": "red",
        })
        assert result["success"] is True
        body = registry._client.post.call_args[1]["json"]
        assert body["rgb_color"] == [255, 0, 0]
        assert "color" not in body

    async def test_kelvin_takes_priority_over_color(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        registry._client = AsyncMock()
        registry._client.post = AsyncMock(return_value=MagicMock(status_code=200))

        await registry.execute_control({
            "entity_id": "light.wiz_1",
            "action": "turn_on",
            "color": "red",
            "color_temp_kelvin": 3000,
        })
        body = registry._client.post.call_args[1]["json"]
        assert "color_temp_kelvin" in body
        assert "rgb_color" not in body

    async def test_brightness_converted_to_brightness_pct(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        registry._client = AsyncMock()
        registry._client.post = AsyncMock(return_value=MagicMock(status_code=200))

        await registry.execute_control({
            "entity_id": "light.wiz_1",
            "action": "turn_on",
            "brightness": 80,
        })
        body = registry._client.post.call_args[1]["json"]
        assert body["brightness_pct"] == 80
        assert "brightness" not in body

    async def test_fan_turn_off_no_light_params(self):
        """Non-light domains: only entity_id in service_data."""
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "fan.vital_100s", "attributes": {"friendly_name": "Vital"}},
        ]
        registry._client = AsyncMock()
        registry._client.post = AsyncMock(return_value=MagicMock(status_code=200))

        result = await registry.execute_control({
            "entity_id": "fan.vital_100s",
            "action": "turn_off",
            "brightness": 50,
        })
        assert result["success"] is True
        body = registry._client.post.call_args[1]["json"]
        assert body == {"entity_id": "fan.vital_100s"}

    async def test_invalid_entity_returns_error(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        result = await registry.execute_control({
            "entity_id": "light.nonexistent",
            "action": "turn_on",
        })
        assert "error" in result

    async def test_http_error_returns_error(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        registry._client = AsyncMock()
        registry._client.post = AsyncMock(return_value=MagicMock(status_code=500))

        result = await registry.execute_control({
            "entity_id": "light.wiz_1",
            "action": "turn_on",
        })
        assert "error" in result


class TestHassRegistryExecuteQuery:
    async def test_get_state_success(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        registry._client = AsyncMock()
        registry._client.get = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={"state": "on", "attributes": {"brightness": 255}}),
        ))

        result = await registry.execute_query({
            "query_type": "get_state",
            "entity_id": "light.wiz_1",
        })
        assert result["state"] == "on"

    async def test_get_state_missing_entity_id_returns_error(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        result = await registry.execute_query({
            "query_type": "get_state",
        })
        assert "error" in result

    async def test_list_entities_returns_all(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ RGBW"}},
            {"entity_id": "fan.vital_100s", "attributes": {"friendly_name": "Vital 100S"}},
        ]
        result = await registry.execute_query({"query_type": "list_entities"})
        assert len(result["entities"]) == 2

    async def test_get_state_invalid_entity_returns_error(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        result = await registry.execute_query({
            "query_type": "get_state",
            "entity_id": "light.nonexistent",
        })
        assert "error" in result


class TestHassRegistryExecuteDispatch:
    """Registry.execute() dispatches to execute_control or execute_query."""

    async def test_dispatch_hass_control(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        registry._client = AsyncMock()
        registry._client.post = AsyncMock(return_value=MagicMock(status_code=200))

        result = await registry.execute("hass_control", {
            "entity_id": "light.wiz_1",
            "action": "turn_on",
        })
        assert result["success"] is True

    async def test_dispatch_hass_query(self):
        registry = HassRegistry("http://hass:8123", "token")
        registry._entities = [
            {"entity_id": "light.wiz_1", "attributes": {"friendly_name": "WiZ"}},
        ]
        result = await registry.execute("hass_query", {"query_type": "list_entities"})
        assert "entities" in result

    async def test_dispatch_unknown_returns_error(self):
        registry = HassRegistry("http://hass:8123", "token")
        result = await registry.execute("unknown", {})
        assert "error" in result
