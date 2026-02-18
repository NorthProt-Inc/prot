"""Backward compatibility facade — use prot.logging instead."""

from prot.logging import (  # noqa: F401
    get_logger,
    setup_logging,
    start_turn,
    elapsed_ms,
    reset_turn,
    logged,
    StructuredLogger,
    SmartFormatter,
    PlainFormatter,
    JsonFormatter,
)
