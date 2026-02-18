"""Tests for logging constants module."""

from prot.logging.constants import (
    MODULE_MAP,
    LEVEL_COLORS,
    RESET,
    DIM,
)


class TestModuleMap:
    def test_contains_core_modules(self):
        for mod in ("pipeline", "stt", "llm", "tts", "playback", "vad", "memory"):
            assert mod in MODULE_MAP

    def test_entries_have_abbrev_and_color(self):
        for mod, (abbrev, color) in MODULE_MAP.items():
            assert len(abbrev) <= 3
            assert color.startswith("\033[")


class TestAnsiCodes:
    def test_reset_code(self):
        assert RESET == "\033[0m"

    def test_dim_code(self):
        assert DIM == "\033[2m"

    def test_level_colors_cover_all_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            assert level in LEVEL_COLORS
