"""Backward compatibility facade — use prot.logging instead."""

from prot.logging import (  # noqa: F401
    get_logger,
    setup_logging,
    start_turn,
    reset_turn,
)
