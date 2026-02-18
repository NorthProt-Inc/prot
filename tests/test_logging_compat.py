"""Tests for backward compatibility via prot.log facade."""

from prot.log import (
    get_logger,
    setup_logging,
    start_turn,
    elapsed_ms,
    reset_turn,
    StructuredLogger,
    SmartFormatter,
)


class TestBackwardCompat:
    def test_get_logger_works(self):
        logger = get_logger("test.compat")
        assert isinstance(logger, StructuredLogger)

    def test_setup_logging_callable(self):
        assert callable(setup_logging)

    def test_turn_tracking_works(self):
        start_turn()
        ms = elapsed_ms()
        assert ms is not None
        reset_turn()
        assert elapsed_ms() is None

    def test_formatter_importable(self):
        assert SmartFormatter is not None
