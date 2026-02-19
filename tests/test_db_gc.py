"""Unit tests for scripts/db_gc.py pure functions (no DB required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load db_gc module from scripts/ (not a package)
_spec = importlib.util.spec_from_file_location(
    "db_gc", Path(__file__).resolve().parent.parent / "scripts" / "db_gc.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

normalize_name = _mod.normalize_name
normalize_relation_type = _mod.normalize_relation_type
dedup_description = _mod.dedup_description
CANONICAL_RELATION_TYPES = _mod.CANONICAL_RELATION_TYPES


class TestNormalizeName:
    def test_lowercase(self):
        assert normalize_name("Claude") == "claude"

    def test_underscores_to_spaces(self):
        assert normalize_name("o4_mini_deep_research") == "o4 mini deep research"

    def test_hyphens_to_spaces(self):
        assert normalize_name("o4-mini-deep-research") == "o4 mini deep research"

    def test_parens_to_spaces(self):
        assert normalize_name("Claude(Opus)") == "claude opus"

    def test_slash_to_space(self):
        assert normalize_name("src/prot") == "src prot"

    def test_collapse_whitespace(self):
        assert normalize_name("  hello   world  ") == "hello world"

    def test_unicode_nfc(self):
        # Composed vs decomposed form should match
        composed = "\u00e9"  # é (NFC)
        decomposed = "e\u0301"  # e + combining acute (NFD)
        assert normalize_name(composed) == normalize_name(decomposed)

    def test_korean_passthrough(self):
        assert normalize_name("자동 Opus 선택") == "자동 opus 선택"

    def test_mixed_separators(self):
        assert normalize_name("foo-bar_baz (qux)") == "foo bar baz qux"

    def test_empty_string(self):
        assert normalize_name("") == ""


class TestNormalizeRelationType:
    def test_canonical_passthrough(self):
        for canonical in CANONICAL_RELATION_TYPES:
            assert normalize_relation_type(canonical) == canonical

    def test_korean_causes(self):
        assert normalize_relation_type("원인") == "causes"
        assert normalize_relation_type("유발") == "causes"
        assert normalize_relation_type("영향") == "causes"

    def test_korean_related(self):
        assert normalize_relation_type("연관") == "related_to"
        assert normalize_relation_type("관련") == "related_to"
        assert normalize_relation_type("연결") == "related_to"

    def test_korean_uses(self):
        assert normalize_relation_type("사용") == "uses"
        assert normalize_relation_type("활용") == "uses"
        assert normalize_relation_type("적용") == "uses"

    def test_korean_contains(self):
        assert normalize_relation_type("포함") == "contains"
        assert normalize_relation_type("구성요소") == "contains"

    def test_korean_triggers(self):
        assert normalize_relation_type("트리거") == "triggers"
        assert normalize_relation_type("발생") == "triggers"
        assert normalize_relation_type("촉발") == "triggers"

    def test_korean_misc(self):
        assert normalize_relation_type("선호") == "prefers"
        assert normalize_relation_type("대안") == "alternative_to"
        assert normalize_relation_type("비유") == "similar_to"
        assert normalize_relation_type("대체") == "replaces"
        assert normalize_relation_type("구현") == "produces"
        assert normalize_relation_type("요청") == "requires"
        assert normalize_relation_type("선행") == "precedes"

    def test_english_variants(self):
        assert normalize_relation_type("caused") == "causes"
        assert normalize_relation_type("caused_by") == "causes"
        assert normalize_relation_type("affects") == "causes"
        assert normalize_relation_type("includes") == "contains"
        assert normalize_relation_type("utilizes") == "uses"
        assert normalize_relation_type("resolves") == "improves"
        assert normalize_relation_type("replacement") == "replaces"
        assert normalize_relation_type("implements") == "produces"

    def test_case_insensitive(self):
        assert normalize_relation_type("CAUSES") == "causes"
        assert normalize_relation_type("Related_To") == "related_to"

    def test_hyphen_to_underscore(self):
        assert normalize_relation_type("related-to") == "related_to"
        assert normalize_relation_type("part-of") == "part_of"

    def test_fallback_unknown(self):
        assert normalize_relation_type("some_weird_type") == "related_to"
        assert normalize_relation_type("xyz123") == "related_to"

    def test_whitespace_stripped(self):
        assert normalize_relation_type("  causes  ") == "causes"
        assert normalize_relation_type("  원인  ") == "causes"


class TestDedupDescription:
    def test_no_newline(self):
        assert dedup_description("simple text") == "simple text"

    def test_duplicate_lines(self):
        assert dedup_description("line1\nline2\nline1") == "line1\nline2"

    def test_preserves_order(self):
        assert dedup_description("b\na\nb\nc\na") == "b\na\nc"

    def test_empty_lines_preserved(self):
        # Empty lines are not duplicated in this case
        assert dedup_description("a\n\nb\n\nc") == "a\n\nb\nc"

    def test_all_same(self):
        assert dedup_description("x\nx\nx") == "x"

    def test_empty_string(self):
        assert dedup_description("") == ""
