"""Structured logging for prot voice pipeline.

Console-only colored output with module abbreviations, k=v pairs,
and per-turn elapsed tracking. Designed for journalctl consumption.
"""

from __future__ import annotations

import logging
import os
import time
from contextvars import ContextVar

# ---------------------------------------------------------------------------
# Per-turn elapsed timer (lightweight request_tracker alternative)
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
# Module color / abbreviation map
# ---------------------------------------------------------------------------

_MODULE_MAP: dict[str, tuple[str, str]] = {
    "pipeline":   ("PIP", "\033[96m"),
    "stt":        ("STT", "\033[94m"),
    "llm":        ("LLM", "\033[92m"),
    "tts":        ("TTS", "\033[38;5;208m"),
    "playback":   ("PLY", "\033[93m"),
    "vad":        ("VAD", "\033[91m"),
    "memory":     ("MEM", "\033[95m"),
    "graphrag":   ("RAG", "\033[38;5;39m"),
    "embeddings": ("EMB", "\033[38;5;147m"),
    "db":         ("DB",  "\033[90m"),
    "context":    ("CTX", "\033[96m"),
    "audio":      ("AUD", "\033[97m"),
    "app":        ("APP", "\033[96m"),
    "config":     ("CFG", "\033[96m"),
}

_RESET = "\033[0m"
_DIM = "\033[2m"

_LEVEL_COLORS: dict[str, str] = {
    "DEBUG":    "\033[37m",
    "INFO":     "\033[97m",
    "WARNING":  "\033[93m",
    "ERROR":    "\033[91m",
    "CRITICAL": "\033[91;1m",
}


def _module_key(name: str) -> str:
    """Extract last dotted segment: 'prot.pipeline' -> 'pipeline'."""
    return name.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# SmartFormatter — colored console output with k=v pairs
# ---------------------------------------------------------------------------

class SmartFormatter(logging.Formatter):
    """Console formatter: ``HH:MM:SS.mmm  LEVEL [ABR|module] msg | k=v +elapsed``"""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        ms = int(record.created * 1000) % 1000
        timestamp = f"{ts}.{ms:03d}"

        mod = _module_key(record.name)
        abbrev, color = _MODULE_MAP.get(mod, (mod[:3].upper(), "\033[37m"))
        level_color = _LEVEL_COLORS.get(record.levelname, "")

        # Format k=v extras
        extra_data: dict = getattr(record, "extra_data", {})
        kv_parts = [f"{k}={v}" for k, v in extra_data.items()]

        # Append elapsed if a turn is active
        ms_elapsed = elapsed_ms()
        if ms_elapsed is not None:
            if ms_elapsed >= 1000:
                kv_parts.append(f"+{ms_elapsed / 1000:.1f}s")
            else:
                kv_parts.append(f"+{ms_elapsed}ms")

        kv_str = f" {_DIM}| {' '.join(kv_parts)}{_RESET}" if kv_parts else ""

        msg = record.getMessage()

        line = (
            f"{_DIM}{timestamp}{_RESET}  "
            f"{level_color}{record.levelname:<5}{_RESET} "
            f"[{color}{abbrev}{_RESET}|{color}{mod:<10}{_RESET}] "
            f"{msg}{kv_str}"
        )

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line += "\n" + record.exc_text

        return line


# ---------------------------------------------------------------------------
# StructuredLogger — stdlib Logger wrapper with **kwargs -> extra_data
# ---------------------------------------------------------------------------

class StructuredLogger:
    """Thin wrapper adding ``**kwargs`` as structured k=v pairs to log records."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, args: tuple, kwargs: dict) -> None:
        exc_info = kwargs.pop("exc_info", None)
        extra_data = kwargs
        self._logger.log(
            level, msg, *args,
            exc_info=exc_info,
            extra={"extra_data": extra_data},
        )

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


# ---------------------------------------------------------------------------
# Logger factory + setup
# ---------------------------------------------------------------------------

_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a StructuredLogger for the given module name."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(logging.getLogger(name))
    return _loggers[name]


def setup_logging(level: str | None = None) -> None:
    """Configure root logger with SmartFormatter. Call once at startup.

    Level resolution: explicit arg > LOG_LEVEL env > config default.
    """
    from prot.config import settings

    resolved = level or os.environ.get("LOG_LEVEL") or settings.log_level
    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(SmartFormatter())
    root.addHandler(handler)