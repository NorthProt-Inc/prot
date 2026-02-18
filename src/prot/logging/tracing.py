"""Function tracing decorator with call-depth visualization.

Provides @logged() decorator for zero-cost entry/exit tracing,
slow-path detection, and optional argument/result logging with
automatic secret redaction.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import re
import time
from contextvars import ContextVar
from typing import Any

from prot.logging.structured_logger import get_logger

# ---------------------------------------------------------------------------
# Call-depth tracking
# ---------------------------------------------------------------------------

_call_depth: ContextVar[int] = ContextVar("call_depth", default=0)

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_RE = re.compile(
    r"(password|token|secret|key|auth|credential|bearer|api_key)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Value formatting utilities
# ---------------------------------------------------------------------------


def _fmt_val(value: Any, max_len: int = 80) -> str:
    """Format a value for log output with truncation and type summaries."""
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, (list, tuple)):
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, dict):
        return f"<dict len={len(value)}>"
    s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _fmt_args(
    func: Any,
    args: tuple,
    kwargs: dict,
    *,
    redact: bool = True,
    max_val_len: int = 80,
) -> str:
    """Format function arguments, skipping self/cls and redacting secrets."""
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
        if redact and _SECRET_RE.search(name):
            parts.append(f"{name}=***")
        else:
            parts.append(f"{name}={_fmt_val(val, max_val_len)}")

    for name, val in kwargs.items():
        if redact and _SECRET_RE.search(name):
            parts.append(f"{name}=***")
        else:
            parts.append(f"{name}={_fmt_val(val, max_val_len)}")

    return ", ".join(parts)


def _fmt_result(value: Any, *, redact: bool = True, max_val_len: int = 80) -> str:
    """Format a return value for log output."""
    return _fmt_val(value, max_val_len)


def _fmt_time(elapsed_s: float) -> str:
    """Format elapsed seconds as human-readable string."""
    ms = elapsed_s * 1000
    if ms >= 1000:
        return f"+{ms / 1000:.1f}s"
    return f"+{ms:.1f}ms"


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _trace_sync(
    func: Any,
    logger: Any,
    level: int,
    log_args: bool,
    log_result: bool,
    slow_ms: float,
    redact: bool,
    max_val_len: int,
) -> Any:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not logger.isEnabledFor(level):
            return func(*args, **kwargs)

        depth = _call_depth.get()
        _call_depth.set(depth + 1)
        name = func.__qualname__

        entry_msg = f"-> {name}"
        if log_args:
            entry_msg += f"({_fmt_args(func, args, kwargs, redact=redact, max_val_len=max_val_len)})"
        logger._log(level, entry_msg, (), {"_depth": depth, "_trace_dir": "entry"})

        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            elapsed_str = _fmt_time(elapsed)

            if slow_ms and elapsed * 1000 > slow_ms:
                exit_msg = f"!! {name} SLOW"
                if log_result:
                    exit_msg += f" => {_fmt_result(result, redact=redact, max_val_len=max_val_len)}"
                logger._log(
                    logging.WARNING, exit_msg, (),
                    {"_depth": depth, "_trace_dir": "slow", "_elapsed": elapsed_str},
                )
            else:
                exit_msg = f"<- {name}"
                if log_result:
                    exit_msg += f" => {_fmt_result(result, redact=redact, max_val_len=max_val_len)}"
                logger._log(
                    level, exit_msg, (),
                    {"_depth": depth, "_trace_dir": "exit", "_elapsed": elapsed_str},
                )
            return result
        except Exception:
            elapsed = time.perf_counter() - t0
            logger._log(
                logging.ERROR, f"!! {name} FAILED", (),
                {"_depth": depth, "_trace_dir": "error", "_elapsed": _fmt_time(elapsed)},
            )
            raise
        finally:
            _call_depth.set(depth)

    return wrapper


def _trace_async(
    func: Any,
    logger: Any,
    level: int,
    log_args: bool,
    log_result: bool,
    slow_ms: float,
    redact: bool,
    max_val_len: int,
) -> Any:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not logger.isEnabledFor(level):
            return await func(*args, **kwargs)

        depth = _call_depth.get()
        _call_depth.set(depth + 1)
        name = func.__qualname__

        entry_msg = f"-> {name}"
        if log_args:
            entry_msg += f"({_fmt_args(func, args, kwargs, redact=redact, max_val_len=max_val_len)})"
        logger._log(level, entry_msg, (), {"_depth": depth, "_trace_dir": "entry"})

        t0 = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            elapsed_str = _fmt_time(elapsed)

            if slow_ms and elapsed * 1000 > slow_ms:
                exit_msg = f"!! {name} SLOW"
                if log_result:
                    exit_msg += f" => {_fmt_result(result, redact=redact, max_val_len=max_val_len)}"
                logger._log(
                    logging.WARNING, exit_msg, (),
                    {"_depth": depth, "_trace_dir": "slow", "_elapsed": elapsed_str},
                )
            else:
                exit_msg = f"<- {name}"
                if log_result:
                    exit_msg += f" => {_fmt_result(result, redact=redact, max_val_len=max_val_len)}"
                logger._log(
                    level, exit_msg, (),
                    {"_depth": depth, "_trace_dir": "exit", "_elapsed": elapsed_str},
                )
            return result
        except Exception:
            elapsed = time.perf_counter() - t0
            logger._log(
                logging.ERROR, f"!! {name} FAILED", (),
                {"_depth": depth, "_trace_dir": "error", "_elapsed": _fmt_time(elapsed)},
            )
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
    redact: bool,
    max_val_len: int,
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

        entry_msg = f"-> {name}"
        if log_args:
            entry_msg += f"({_fmt_args(func, args, kwargs, redact=redact, max_val_len=max_val_len)})"
        logger._log(level, entry_msg, (), {"_depth": depth, "_trace_dir": "entry"})

        t0 = time.perf_counter()
        try:
            async for item in func(*args, **kwargs):
                yield item
            elapsed = time.perf_counter() - t0
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
        except Exception:
            elapsed = time.perf_counter() - t0
            logger._log(
                logging.ERROR, f"!! {name} FAILED", (),
                {"_depth": depth, "_trace_dir": "error", "_elapsed": _fmt_time(elapsed)},
            )
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
    log_result: bool = False,
    slow_ms: float = 0,
    redact: bool = True,
    max_val_len: int = 80,
):
    """Zero-cost function tracing decorator.

    When the configured log level is disabled, the original function
    is called directly with no overhead.

    Args:
        level: Log level for entry/exit messages (default DEBUG).
        log_args: Log function arguments on entry.
        log_result: Log return value on exit (not applicable to async generators).
        slow_ms: Threshold in ms; exceeding triggers WARNING. 0 disables.
        redact: Auto-mask arguments whose names match secret patterns.
        max_val_len: Max repr length for logged values.
    """

    def decorator(func):
        # Resolve logger from the function's module
        module = getattr(func, "__module__", None) or __name__
        _logger = get_logger(module)

        if inspect.isasyncgenfunction(func):
            return _trace_async_gen(
                func, _logger, level, log_args, slow_ms, redact, max_val_len,
            )
        elif asyncio.iscoroutinefunction(func):
            return _trace_async(
                func, _logger, level, log_args, log_result, slow_ms, redact, max_val_len,
            )
        else:
            return _trace_sync(
                func, _logger, level, log_args, log_result, slow_ms, redact, max_val_len,
            )

    return decorator
