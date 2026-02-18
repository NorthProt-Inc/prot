"""Tests for logging formatters."""

import logging

from prot.logging.formatters import SmartFormatter, PlainFormatter


def _make_record(msg="test message", level=logging.INFO, name="prot.pipeline", **extra):
    """Helper to create a LogRecord with extra_data."""
    record = logging.LogRecord(
        name=name, level=level, pathname="", lineno=0,
        msg=msg, args=(), exc_info=None,
    )
    record.extra_data = extra
    return record


class TestSmartFormatter:
    def test_output_contains_module_abbrev(self):
        fmt = SmartFormatter()
        record = _make_record(name="prot.pipeline")
        line = fmt.format(record)
        assert "PIP" in line
        assert "pipeline" in line

    def test_output_contains_kv_pairs(self):
        fmt = SmartFormatter()
        record = _make_record(port=8000, env="prod")
        line = fmt.format(record)
        assert "port=8000" in line
        assert "env=prod" in line

    def test_output_contains_level(self):
        fmt = SmartFormatter()
        record = _make_record(level=logging.WARNING)
        line = fmt.format(record)
        assert "WARNI" in line or "WARNING" in line


class TestPlainFormatter:
    def test_no_ansi_codes(self):
        fmt = PlainFormatter()
        record = _make_record(name="prot.stt")
        line = fmt.format(record)
        assert "\033[" not in line

    def test_contains_full_date(self):
        fmt = PlainFormatter()
        record = _make_record()
        line = fmt.format(record)
        assert "-" in line.split(" ")[0]

    def test_contains_kv_pairs(self):
        fmt = PlainFormatter()
        record = _make_record(attempt=3)
        line = fmt.format(record)
        assert "attempt=3" in line
