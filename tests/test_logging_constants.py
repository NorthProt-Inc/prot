"""Tests for logging constants module."""

from prot.logging.constants import (
    MODULE_MAP,
    LEVEL_COLORS,
    ABBREVIATIONS,
    RESET,
    DIM,
    abbreviate,
)


class TestModuleMap:
    def test_contains_core_modules(self):
        for mod in ("pipeline", "stt", "llm", "tts", "playback", "vad", "memory"):
            assert mod in MODULE_MAP

    def test_entries_have_abbrev_and_color(self):
        for mod, (abbrev, color) in MODULE_MAP.items():
            assert len(abbrev) <= 3
            assert color.startswith("\033[")


class TestAbbreviations:
    def test_contains_common_terms(self):
        assert ABBREVIATIONS["request"] == "req"
        assert ABBREVIATIONS["response"] == "res"
        assert ABBREVIATIONS["message"] == "msg"
        assert ABBREVIATIONS["connection"] == "conn"

    def test_abbreviate_replaces_words(self):
        assert abbreviate("request received") == "req recv"

    def test_abbreviate_preserves_unknown(self):
        assert abbreviate("hello world") == "hello world"

    def test_abbreviate_case_insensitive(self):
        result = abbreviate("Request completed")
        assert result == "req done"

    def test_abbreviate_empty_string(self):
        assert abbreviate("") == ""


class TestAnsiCodes:
    def test_reset_code(self):
        assert RESET == "\033[0m"

    def test_dim_code(self):
        assert DIM == "\033[2m"

    def test_level_colors_cover_all_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            assert level in LEVEL_COLORS
