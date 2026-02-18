"""Tests for async file logging handlers."""

import logging
import time

from prot.logging.handlers import create_async_handler


class TestAsyncHandler:
    def test_creates_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler, listener = create_async_handler(str(log_file))
        listener.start()

        logger = logging.getLogger("test.async_handler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("hello async")

        time.sleep(0.1)
        listener.stop()
        logger.removeHandler(handler)

        assert log_file.exists()
        content = log_file.read_text()
        assert "hello async" in content

    def test_rotates_on_max_bytes(self, tmp_path):
        log_file = tmp_path / "rotate.log"
        handler, listener = create_async_handler(
            str(log_file), max_bytes=100, backup_count=2
        )
        listener.start()

        logger = logging.getLogger("test.rotate")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        for i in range(50):
            logger.info(f"line {i} " + "x" * 50)

        time.sleep(0.2)
        listener.stop()
        logger.removeHandler(handler)

        log_files = list(tmp_path.glob("rotate.log*"))
        assert len(log_files) > 1


class TestErrorHandler:
    def test_only_captures_errors(self, tmp_path):
        log_file = tmp_path / "error.log"
        handler, listener = create_async_handler(
            str(log_file), level=logging.ERROR
        )
        listener.start()

        logger = logging.getLogger("test.error_filter")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)
        listener.stop()
        logger.removeHandler(handler)

        content = log_file.read_text()
        assert "info message" not in content
        assert "warning message" not in content
        assert "error message" in content
        assert "critical message" in content
