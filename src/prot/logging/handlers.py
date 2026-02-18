"""Async file logging handlers using QueueHandler + QueueListener."""

from __future__ import annotations

import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from queue import Queue


def create_async_handler(
    log_path: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    formatter: logging.Formatter | None = None,
) -> tuple[QueueHandler, QueueListener]:
    """Create an async file handler.

    Returns (queue_handler, listener). Caller must call listener.start()
    and listener.stop() at shutdown.
    """
    queue: Queue = Queue(-1)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    if formatter:
        file_handler.setFormatter(formatter)
    else:
        file_handler.setFormatter(logging.Formatter("%(message)s"))
    listener = QueueListener(queue, file_handler, respect_handler_level=True)
    queue_handler = QueueHandler(queue)
    return queue_handler, listener


def create_error_handler(
    log_path: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    formatter: logging.Formatter | None = None,
) -> tuple[QueueHandler, QueueListener]:
    """Create an async error-only file handler (ERROR and CRITICAL)."""
    queue: Queue = Queue(-1)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    file_handler.setLevel(logging.ERROR)
    if formatter:
        file_handler.setFormatter(formatter)
    else:
        file_handler.setFormatter(logging.Formatter("%(message)s"))
    listener = QueueListener(queue, file_handler, respect_handler_level=True)
    queue_handler = QueueHandler(queue)
    return queue_handler, listener
