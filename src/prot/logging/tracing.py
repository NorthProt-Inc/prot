"""Function tracing decorator with call-depth visualization.

Provides @logged() decorator for zero-cost entry/exit tracing
and slow-path detection for async functions and async generators.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from contextvars import ContextVar
from typing import Any

from prot.logging.structured_logger import get_logger

# ---------------------------------------------------------------------------
# Call-depth tracking
# ---------------------------------------------------------------------------

_call_depth: ContextVar[int] = ContextVar("call_depth", default=0)

# ---------------------------------------------------------------------------
# Value formatting utilities
# ---------------------------------------------------------------------------

_MAX_VAL_LEN = 80


def _fmt_val(value: Any) -> str:
    """Format a value for log output with truncation and type summaries."""
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, (list, tuple)):
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, dict):
        return f"<dict len={len(value)}>"
    s = repr(value)
    if len(s) > _MAX_VAL_LEN:
        return s[: _MAX_VAL_LEN - 3] + "..."
    return s


def _fmt_args(func: Any, args: tuple, kwargs: dict) -> str:
    """Format function arguments, skipping self/cls."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Skip self/cls for bound methods
    display_args = args
    display_params = params
    if params and params[0] in ("self", "cls"):
        display_args = args[1:] if args else args
        display_params = params[1:]

    parts: list[str] = []
    for i, val in enumerate(display_args):
        name = display_params[i] if i < len(display_params) else f"arg{i}"
        parts.append(f"{name}={_fmt_val(val)}")

    for name, val in kwargs.items():
        parts.append(f"{name}={_fmt_val(val)}")

    return ", ".join(parts)


def _fmt_time(elapsed_s: float) -> str:
    """Format elapsed seconds as human-readable string."""
    ms = elapsed_s * 1000
    if ms >= 1000:
        return f"+{ms / 1000:.1f}s"
    return f"+{ms:.1f}ms"


# ---------------------------------------------------------------------------
# Shared entry/exit logging helpers
# ---------------------------------------------------------------------------


def _log_entry(
    logger: Any,
    level: int,
    name: str,
    depth: int,
    log_args: bool,
    func: Any,
    args: tuple,
    kwargs: dict,
) -> None:
    """Log function entry with optional argument formatting."""
    entry_msg = f"-> {name}"
    if log_args:
        entry_msg += f"({_fmt_args(func, args, kwargs)})"
    logger._log(level, entry_msg, (), {"_depth": depth, "_trace_dir": "entry"})


def _log_exit(
    logger: Any,
    level: int,
    name: str,
    depth: int,
    elapsed: float,
    slow_ms: float,
) -> None:
    """Log function exit with elapsed time and optional slow-path warning."""
    elapsed_str = _fmt_time(elapsed)
    if slow_ms and elapsed * 1000 > slow_ms:
        logger._log(
            logging.WARNING, f"!! {name} SLOW", (),
            {"_depth": depth, "_trace_dir": "slow", "_elapsed": elapsed_str},
        )
    else:
        logger._log(
            level, f"<- {name}", (),
            {"_depth": depth, "_trace_dir": "exit", "_elapsed": elapsed_str},
        )


def _log_error(
    logger: Any,
    name: str,
    depth: int,
    elapsed: float,
) -> None:
    """Log function failure with elapsed time."""
    logger._log(
        logging.ERROR, f"!! {name} FAILED", (),
        {"_depth": depth, "_trace_dir": "error", "_elapsed": _fmt_time(elapsed)},
    )


# ---------------------------------------------------------------------------
# Trace wrappers
# ---------------------------------------------------------------------------


def _trace_async(
    func: Any,
    logger: Any,
    level: int,
    log_args: bool,
    slow_ms: float,
) -> Any:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not logger.isEnabledFor(level):
            return await func(*args, **kwargs)

        depth = _call_depth.get()
        _call_depth.set(depth + 1)
        name = func.__qualname__

        _log_entry(logger, level, name, depth, log_args, func, args, kwargs)

        t0 = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            _log_exit(logger, level, name, depth, time.perf_counter() - t0, slow_ms)
            return result
        except Exception:
            _log_error(logger, name, depth, time.perf_counter() - t0)
            raise
        finally:
            _call_depth.set(depth)

    return wrapper


def _trace_async_gen(
    func: Any,
    logger: Any,
    level: int,
    log_args: bool,
    slow_ms: float,
) -> Any:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any):
        if not logger.isEnabledFor(level):
            async for item in func(*args, **kwargs):
                yield item
            return

        depth = _call_depth.get()
        _call_depth.set(depth + 1)
        name = func.__qualname__

        _log_entry(logger, level, name, depth, log_args, func, args, kwargs)

        t0 = time.perf_counter()
        try:
            async for item in func(*args, **kwargs):
                yield item
            _log_exit(logger, level, name, depth, time.perf_counter() - t0, slow_ms)
        except Exception:
            _log_error(logger, name, depth, time.perf_counter() - t0)
            raise
        finally:
            _call_depth.set(depth)

    return wrapper


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


def logged(
    *,
    level: int = logging.DEBUG,
    log_args: bool = False,
    slow_ms: float = 0,
):
    """Zero-cost function tracing decorator.

    When the configured log level is disabled, the original function
    is called directly with no overhead.

    Args:
        level: Log level for entry/exit messages (default DEBUG).
        log_args: Log function arguments on entry.
        slow_ms: Threshold in ms; exceeding triggers WARNING. 0 disables.
    """

    def decorator(func):
        # Resolve logger from the function's module
        module = getattr(func, "__module__", None) or __name__
        _logger = get_logger(module)

        if inspect.isasyncgenfunction(func):
            return _trace_async_gen(func, _logger, level, log_args, slow_ms)
        elif asyncio.iscoroutinefunction(func):
            return _trace_async(func, _logger, level, log_args, slow_ms)
        else:
            raise TypeError(
                f"@logged() only supports async functions and async generators, "
                f"got sync function: {func.__qualname__}"
            )

    return decorator
