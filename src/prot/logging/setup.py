"""Logging setup: configure root logger with console + async file handlers."""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path

from prot.logging.formatters import SmartFormatter, PlainFormatter, JsonFormatter
from prot.logging.handlers import create_async_handler, create_error_handler

_listeners: list = []


def setup_logging(
    level: str | None = None,
    log_dir: str = "logs",
) -> None:
    """Configure root logger with console + async file handlers.

    Level resolution: explicit arg > LOG_LEVEL env > config default.
    """
    from prot.config import settings

    resolved = level or os.environ.get("LOG_LEVEL") or settings.log_level
    log_dir = os.environ.get("LOG_DIR", log_dir)
    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    # Stop any existing listeners
    for listener in _listeners:
        listener.stop()
    _listeners.clear()

    # 1. Console handler (colored)
    console = logging.StreamHandler()
    console.setFormatter(SmartFormatter())
    root.addHandler(console)

    # 2. File handlers (async)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 2a. Main log file (rotated, plain text)
    file_handler, file_listener = create_async_handler(
        str(log_path / "prot.log"),
        formatter=PlainFormatter(),
    )
    root.addHandler(file_handler)
    file_listener.start()
    _listeners.append(file_listener)

    # 2b. Error-only log file
    error_handler, error_listener = create_error_handler(
        str(log_path / "prot_error.log"),
        formatter=PlainFormatter(),
    )
    root.addHandler(error_handler)
    error_listener.start()
    _listeners.append(error_listener)

    # 3. Optional JSONL handler
    json_enabled = os.environ.get("LOG_JSON", "").lower() in ("1", "true", "yes")
    if json_enabled:
        json_handler, json_listener = create_async_handler(
            str(log_path / "prot.jsonl"),
            formatter=JsonFormatter(),
        )
        root.addHandler(json_handler)
        json_listener.start()
        _listeners.append(json_listener)

    atexit.register(_shutdown_listeners)


def _shutdown_listeners() -> None:
    for listener in _listeners:
        try:
            listener.stop()
        except Exception:
            pass
    _listeners.clear()
