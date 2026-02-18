"""Tests for StructuredLogger and turn tracking."""

import logging

from prot.logging.structured_logger import (
    StructuredLogger,
    get_logger,
    start_turn,
    elapsed_ms,
    reset_turn,
)


class TestStructuredLogger:
    def test_info_logs_message(self, caplog):
        with caplog.at_level(logging.INFO, logger="test.structured"):
            sl = StructuredLogger(logging.getLogger("test.structured"))
            sl.info("hello")
        assert "hello" in caplog.text

    def test_kwargs_stored_as_extra_data(self):
        inner = logging.getLogger("test.extra")
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)

        sl = StructuredLogger(inner)
        sl.info("msg", port=8000, env="prod")

        assert len(records) == 1
        assert records[0].extra_data == {"port": 8000, "env": "prod"}
        inner.removeHandler(handler)

    def test_exception_includes_exc_info(self):
        inner = logging.getLogger("test.exc")
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)

        sl = StructuredLogger(inner)
        try:
            raise ValueError("boom")
        except ValueError:
            sl.exception("failed")

        assert records[0].exc_info is not None
        inner.removeHandler(handler)


class TestGetLogger:
    def test_returns_structured_logger(self):
        logger = get_logger("test.factory")
        assert isinstance(logger, StructuredLogger)

    def test_same_name_returns_same_instance(self):
        a = get_logger("test.same")
        b = get_logger("test.same")
        assert a is b


class TestTurnTracking:
    def test_elapsed_none_before_start(self):
        reset_turn()
        assert elapsed_ms() is None

    def test_elapsed_returns_int_after_start(self):
        start_turn()
        ms = elapsed_ms()
        assert isinstance(ms, int)
        assert ms >= 0
        reset_turn()

    def test_reset_clears_timer(self):
        start_turn()
        reset_turn()
        assert elapsed_ms() is None
