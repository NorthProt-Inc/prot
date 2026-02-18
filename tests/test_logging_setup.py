"""Tests for logging setup and public API."""

import logging
import time

from prot.logging import (
    get_logger,
    setup_logging,
    start_turn,
    elapsed_ms,
    reset_turn,
    StructuredLogger,
    SmartFormatter,
    PlainFormatter,
)
from prot.logging.setup import _listeners


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "testlogs"
        setup_logging(level="DEBUG", log_dir=str(log_dir))
        assert log_dir.exists()

    def test_creates_prot_log_file(self, tmp_path):
        log_dir = tmp_path / "testlogs2"
        setup_logging(level="DEBUG", log_dir=str(log_dir))
        logger = get_logger("test.setup")
        logger.info("setup test")
        time.sleep(0.2)
        assert (log_dir / "prot.log").exists()

    def test_creates_error_log_file(self, tmp_path):
        log_dir = tmp_path / "testlogs3"
        setup_logging(level="DEBUG", log_dir=str(log_dir))
        logger = get_logger("test.error_setup")
        logger.error("error test")
        time.sleep(0.2)
        assert (log_dir / "prot_error.log").exists()


class TestPublicAPI:
    def test_get_logger_accessible(self):
        logger = get_logger("test.api")
        assert isinstance(logger, StructuredLogger)

    def test_turn_tracking_accessible(self):
        start_turn()
        ms = elapsed_ms()
        assert ms is not None
        reset_turn()
