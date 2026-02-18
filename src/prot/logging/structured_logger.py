"""StructuredLogger wrapper and turn tracking."""

from __future__ import annotations

import logging
import sys
import time
from contextvars import ContextVar

# ---------------------------------------------------------------------------
# Per-turn elapsed timer
# ---------------------------------------------------------------------------

_turn_start: ContextVar[float | None] = ContextVar("turn_start", default=None)


def start_turn() -> None:
    """Mark the beginning of a pipeline turn."""
    _turn_start.set(time.monotonic())


def elapsed_ms() -> int | None:
    """Milliseconds since start_turn(), or None if no active turn."""
    t0 = _turn_start.get()
    return int((time.monotonic() - t0) * 1000) if t0 is not None else None


def reset_turn() -> None:
    """Clear the current turn timer."""
    _turn_start.set(None)


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------

class StructuredLogger:
    """Thin wrapper adding **kwargs as structured k=v pairs to log records."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def name(self) -> str:
        return self._logger.name

    def _log(self, level: int, msg: str, args: tuple, kwargs: dict) -> None:
        if not self._logger.isEnabledFor(level):
            return
        exc_info = kwargs.pop("exc_info", None)
        if exc_info is True:
            exc_info = sys.exc_info()
        extra_data = kwargs
        record = self._logger.makeRecord(
            self._logger.name, level, "", 0, msg, args,
            exc_info=exc_info, extra={"extra_data": extra_data},
        )
        record.extra_data = extra_data
        ms = elapsed_ms()
        if ms is not None:
            record.elapsed_ms = ms
        self._logger.handle(record)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.DEBUG, msg, args, kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.INFO, msg, args, kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.WARNING, msg, args, kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.ERROR, msg, args, kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, args, kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        kwargs["exc_info"] = kwargs.get("exc_info", True)
        self._log(logging.ERROR, msg, args, kwargs)

    def isEnabledFor(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a StructuredLogger for the given module name."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(logging.getLogger(name))
    return _loggers[name]
