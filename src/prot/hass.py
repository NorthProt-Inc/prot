"""Home Assistant integration — auto-discovery, 2-tool split, enum-constrained schemas."""

from __future__ import annotations

import colorsys
import re

import httpx

from prot.log import get_logger

logger = get_logger(__name__)

_HASS_TIMEOUT = httpx.Timeout(10.0)
_ALLOWED_DOMAINS = {"light", "fan", "weather", "sensor", "switch", "climate"}

_COLOR_NAMES: dict[str, list[int]] = {
    # English
    "red": [255, 0, 0],
    "green": [0, 128, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "pink": [255, 192, 203],
    "white": [255, 255, 255],
    "warm": [255, 180, 107],
    "cool": [166, 209, 255],
    # Korean
    "빨강": [255, 0, 0],
    "파랑": [0, 0, 255],
    "초록": [0, 128, 0],
    "노랑": [255, 255, 0],
    "분홍": [255, 192, 203],
    "보라": [128, 0, 128],
    "주황": [255, 165, 0],
    "하양": [255, 255, 255],
    "흰색": [255, 255, 255],
}

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")
_RGB_RE = re.compile(r"^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$")
_HSL_RE = re.compile(r"^(?:hsl\()?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?$")


def parse_color(input: str) -> list[int] | None:
    """Parse color string to [R, G, B]. Returns None if unrecognized."""
    if not input:
        return None
    s = input.strip()

    # Named colors (case-insensitive for English)
    lower = s.lower()
    if lower in _COLOR_NAMES:
        return list(_COLOR_NAMES[lower])
    if s in _COLOR_NAMES:
        return list(_COLOR_NAMES[s])

    # Hex
    m = _HEX_RE.match(s)
    if m:
        h = m.group(1)
        return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]

    # RGB
    m = _RGB_RE.match(s)
    if m:
        return [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    # HSL
    m = _HSL_RE.match(s)
    if m:
        h, s_pct, l_pct = int(m.group(1)), int(m.group(2)), int(m.group(3))
        r, g, b = colorsys.hls_to_rgb(h / 360.0, l_pct / 100.0, s_pct / 100.0)
        return [round(r * 255), round(g * 255), round(b * 255)]

    return None


class HassRegistry:
    """Auto-discovers HASS entities and builds enum-constrained tool schemas."""

    def __init__(self, url: str, token: str) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._client = httpx.AsyncClient(timeout=_HASS_TIMEOUT)
        self._entities: list[dict] = []

    async def discover(self) -> None:
        """Fetch entities from HASS API. Called once at startup."""
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            r = await self._client.get(f"{self._url}/api/states", headers=headers)
            if r.status_code != 200:
                logger.warning("HASS discovery failed", status=r.status_code)
                self._entities = []
                return
            all_entities = r.json()
            self._entities = [
                e for e in all_entities
                if e["entity_id"].split(".", 1)[0] in _ALLOWED_DOMAINS
            ]
            logger.info("HASS discovered", count=len(self._entities))
        except Exception:
            logger.warning("HASS discovery failed — no HASS tools", exc_info=True)
            self._entities = []

    def build_tool_schemas(self) -> list[dict]:
        """Build hass_control + hass_query tool definitions with entity enums."""
        if not self._entities:
            return []

        entity_ids = [e["entity_id"] for e in self._entities]
        entity_list = ", ".join(
            f'{e["entity_id"]} ({e["attributes"].get("friendly_name", "")})'
            for e in self._entities
        )

        hass_control: dict = {
            "name": "hass_control",
            "description": (
                f"Control Home Assistant device.\n"
                f"Available: {entity_list}\n"
                f"color and color_temp_kelvin are mutually exclusive; "
                f"color_temp_kelvin takes priority."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "enum": entity_ids},
                    "action": {
                        "type": "string",
                        "enum": ["turn_on", "turn_off", "toggle"],
                    },
                    "brightness": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Brightness percentage (lights only)",
                    },
                    "color": {
                        "type": "string",
                        "description": "Color name (red, 빨강, warm, #FF0000) — lights only",
                    },
                    "color_temp_kelvin": {
                        "type": "integer",
                        "minimum": 2200,
                        "maximum": 6500,
                        "description": "Color temperature in Kelvin — lights only",
                    },
                },
                "required": ["entity_id", "action"],
            },
        }
        hass_query: dict = {
            "name": "hass_query",
            "description": "Query Home Assistant entity states.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "enum": entity_ids},
                    "query_type": {
                        "type": "string",
                        "enum": ["get_state", "list_entities"],
                    },
                },
                "required": ["query_type"],
            },
            "cache_control": {"type": "ephemeral"},
        }
        return [hass_control, hass_query]

    async def close(self) -> None:
        """Close the httpx client."""
        await self._client.aclose()
