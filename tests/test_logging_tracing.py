"""Tests for prot.logging.tracing -- @logged decorator."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from prot.logging.tracing import (
    _call_depth,
    _fmt_args,
    _fmt_val,
    logged,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_logger(monkeypatch):
    """Return (logger_mock, records) that captures _log calls."""
    records: list[dict] = []

    class FakeLogger:
        def isEnabledFor(self, level):
            return True

        def _log(self, level, msg, args, kwargs):
            records.append({"level": level, "msg": msg, "extra": dict(kwargs)})

    fake = FakeLogger()
    # Patch get_logger to return our fake
    monkeypatch.setattr("prot.logging.tracing.get_logger", lambda name: fake)
    return fake, records


# ---------------------------------------------------------------------------
# _fmt_val tests
# ---------------------------------------------------------------------------

class TestFmtVal:
    def test_bytes_summary(self):
        assert _fmt_val(b"hello") == "<bytes len=5>"

    def test_list_summary(self):
        assert _fmt_val([1, 2, 3]) == "<list len=3>"

    def test_dict_summary(self):
        assert _fmt_val({"a": 1}) == "<dict len=1>"

    def test_truncation(self):
        long_str = "x" * 200
        result = _fmt_val(long_str)
        assert len(result) == 80
        assert result.endswith("...")

    def test_short_value(self):
        assert _fmt_val(42) == "42"


# ---------------------------------------------------------------------------
# _fmt_args tests
# ---------------------------------------------------------------------------

class TestFmtArgs:
    def test_skip_self(self):
        class Foo:
            def bar(self, x, y):
                pass
        result = _fmt_args(Foo.bar, (Foo(), 1, 2), {})
        assert "x=1" in result
        assert "y=2" in result
        assert "self" not in result

    def test_skip_cls(self):
        class Foo:
            @classmethod
            def bar(cls, x):
                pass
        result = _fmt_args(Foo.__dict__["bar"].__func__, (Foo, 1), {})
        assert "x=1" in result
        assert "cls" not in result

    def test_kwargs(self):
        def fn():
            pass
        result = _fmt_args(fn, (), {"name": "bob"})
        assert "bob" in result


# ---------------------------------------------------------------------------
# @logged on async coroutines
# ---------------------------------------------------------------------------

class TestLoggedAsync:
    async def test_async_basic(self, monkeypatch):
        _, records = _capture_logger(monkeypatch)

        @logged()
        async def fetch():
            return "data"

        result = await fetch()
        assert result == "data"
        assert any("->" in r["msg"] for r in records)
        assert any("<-" in r["msg"] for r in records)

    async def test_async_exception_restores_depth(self, monkeypatch):
        _, records = _capture_logger(monkeypatch)

        @logged()
        async def fail():
            raise RuntimeError("async boom")

        assert _call_depth.get() == 0
        with pytest.raises(RuntimeError, match="async boom"):
            await fail()
        assert _call_depth.get() == 0
        assert any("FAILED" in r["msg"] for r in records)

    async def test_async_slow_warning(self, monkeypatch):
        _, records = _capture_logger(monkeypatch)

        @logged(slow_ms=1)
        async def slow_fn():
            await asyncio.sleep(0.01)
            return "done"

        await slow_fn()
        assert any(
            r["level"] == logging.WARNING and "SLOW" in r["msg"]
            for r in records
        )


# ---------------------------------------------------------------------------
# @logged on async generators
# ---------------------------------------------------------------------------

class TestLoggedAsyncGen:
    async def test_async_gen_basic(self, monkeypatch):
        _, records = _capture_logger(monkeypatch)

        @logged()
        async def gen():
            yield 1
            yield 2
            yield 3

        items = [item async for item in gen()]
        assert items == [1, 2, 3]
        assert any("->" in r["msg"] for r in records)
        assert any("<-" in r["msg"] for r in records)

    async def test_async_gen_exception_restores_depth(self, monkeypatch):
        _, records = _capture_logger(monkeypatch)

        @logged()
        async def fail_gen():
            yield 1
            raise ValueError("gen boom")

        assert _call_depth.get() == 0
        with pytest.raises(ValueError, match="gen boom"):
            async for _ in fail_gen():
                pass
        assert _call_depth.get() == 0
        assert any("FAILED" in r["msg"] for r in records)

    async def test_async_gen_slow_warning(self, monkeypatch):
        _, records = _capture_logger(monkeypatch)

        @logged(slow_ms=1)
        async def slow_gen():
            await asyncio.sleep(0.01)
            yield "done"

        async for _ in slow_gen():
            pass
        assert any(
            r["level"] == logging.WARNING and "SLOW" in r["msg"]
            for r in records
        )


# ---------------------------------------------------------------------------
# Call-depth tracking
# ---------------------------------------------------------------------------

class TestCallDepth:
    async def test_nested_async_depth(self, monkeypatch):
        _, records = _capture_logger(monkeypatch)

        @logged()
        async def outer():
            await inner()

        @logged()
        async def inner():
            pass

        await outer()
        depths = [r["extra"].get("_depth") for r in records if "->" in r["msg"]]
        assert depths == [0, 1]


# ---------------------------------------------------------------------------
# Zero-cost when level disabled
# ---------------------------------------------------------------------------

class TestZeroCost:
    async def test_disabled_level_async(self, monkeypatch):
        call_count = 0

        class DisabledLogger:
            def isEnabledFor(self, level):
                return False

            def _log(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1

        monkeypatch.setattr("prot.logging.tracing.get_logger", lambda name: DisabledLogger())

        @logged()
        async def fetch():
            return "data"

        result = await fetch()
        assert result == "data"
        assert call_count == 0

    async def test_disabled_level_async_gen(self, monkeypatch):
        call_count = 0

        class DisabledLogger:
            def isEnabledFor(self, level):
                return False

            def _log(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1

        monkeypatch.setattr("prot.logging.tracing.get_logger", lambda name: DisabledLogger())

        @logged()
        async def gen():
            yield 1
            yield 2

        items = [item async for item in gen()]
        assert items == [1, 2]
        assert call_count == 0


# ---------------------------------------------------------------------------
# Method self-skip
# ---------------------------------------------------------------------------

class TestMethodSelfSkip:
    async def test_self_not_logged(self, monkeypatch):
        _, records = _capture_logger(monkeypatch)

        class MyService:
            @logged(log_args=True)
            async def do_work(self, x):
                return x * 2

        svc = MyService()
        result = await svc.do_work(5)
        assert result == 10
        entry = [r for r in records if "->" in r["msg"]][0]
        # Extract args portion from "-> qualname(args)"
        args_part = entry["msg"].split("(", 1)[1] if "(" in entry["msg"] else ""
        assert "self=" not in args_part
        assert "x=5" in args_part


# ---------------------------------------------------------------------------
# Sync function rejection
# ---------------------------------------------------------------------------

class TestSyncRejection:
    def test_sync_raises_type_error(self, monkeypatch):
        _capture_logger(monkeypatch)

        with pytest.raises(TypeError, match="only supports async"):
            @logged()
            def sync_fn():
                return 42
