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
    level: int | None = None,
) -> tuple[QueueHandler, QueueListener]:
    """Create an async file handler.

    Returns (queue_handler, listener). Caller must call listener.start()
    and listener.stop() at shutdown.

    Pass level=logging.ERROR to create an error-only handler.
    """
    queue: Queue = Queue(-1)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    if level is not None:
        file_handler.setLevel(level)
    if formatter:
        file_handler.setFormatter(formatter)
    else:
        file_handler.setFormatter(logging.Formatter("%(message)s"))
    listener = QueueListener(queue, file_handler, respect_handler_level=True)
    queue_handler = QueueHandler(queue)
    return queue_handler, listener
