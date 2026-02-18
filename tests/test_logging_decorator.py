"""Tests for @logged decorator."""

import logging

import pytest

from prot.logging.decorator import logged
from prot.logging.structured_logger import get_logger


class TestLoggedDecorator:
    def test_sync_function_entry_exit(self, caplog):
        @logged()
        def add(a, b):
            return a + b

        with caplog.at_level(logging.DEBUG):
            result = add(1, 2)
        assert result == 3
        messages = caplog.text
        assert "\u2192" in messages or "→" in messages
        assert "\u2190" in messages or "←" in messages

    @pytest.mark.asyncio
    async def test_async_function_entry_exit(self, caplog):
        @logged()
        async def fetch(url):
            return "data"

        with caplog.at_level(logging.DEBUG):
            result = await fetch("http://example.com")
        assert result == "data"
        messages = caplog.text
        assert "\u2192" in messages or "→" in messages

    def test_exception_logged_with_marker(self, caplog):
        @logged()
        def fail():
            raise ValueError("boom")

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError, match="boom"):
                fail()
        messages = caplog.text
        assert "\u2717" in messages or "✗" in messages

    def test_log_args_includes_arguments(self, caplog):
        @logged(log_args=True)
        def greet(name):
            return f"hi {name}"

        with caplog.at_level(logging.DEBUG):
            greet("alice")
        # extra_data is stored on records, not in caplog.text (pytest formatter)
        entry_record = [r for r in caplog.records if "\u2192" in r.getMessage()][0]
        assert entry_record.extra_data["name"] == "alice"

    def test_log_result_includes_return_value(self, caplog):
        @logged(log_result=True)
        def double(x):
            return x * 2

        with caplog.at_level(logging.DEBUG):
            double(5)
        exit_record = [r for r in caplog.records if "\u2190" in r.getMessage()][0]
        assert exit_record.extra_data["result"] == 10

    def test_custom_level(self, caplog):
        @logged(level=logging.WARNING)
        def important():
            return 42

        with caplog.at_level(logging.WARNING):
            important()
        assert "important" in caplog.text
