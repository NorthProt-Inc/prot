"""prot structured logging package.

Public API:
    get_logger      — Get a StructuredLogger for a module
    setup_logging   — Configure root logger (call once at startup)
    start_turn      — Mark beginning of pipeline turn
    elapsed_ms      — Get ms since start_turn()
    reset_turn      — Clear turn timer
    logged          — Decorator for auto entry/exit logging
    StructuredLogger, SmartFormatter, PlainFormatter, JsonFormatter
"""

from prot.logging.structured_logger import (
    StructuredLogger,
    get_logger,
    start_turn,
    elapsed_ms,
    reset_turn,
)
from prot.logging.formatters import SmartFormatter, PlainFormatter, JsonFormatter
from prot.logging.decorator import logged
from prot.logging.setup import setup_logging

__all__ = [
    "get_logger",
    "setup_logging",
    "start_turn",
    "elapsed_ms",
    "reset_turn",
    "logged",
    "StructuredLogger",
    "SmartFormatter",
    "PlainFormatter",
    "JsonFormatter",
]
