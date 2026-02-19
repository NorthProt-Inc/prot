"""Home Assistant integration — auto-discovery, 2-tool split, enum-constrained schemas."""

from __future__ import annotations

import colorsys
import re

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
