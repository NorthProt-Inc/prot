"""@logged decorator for automatic function entry/exit logging."""

from __future__ import annotations

import functools
import inspect
import logging
import time
from typing import Any

from prot.logging.structured_logger import get_logger


def logged(
    level: int = logging.DEBUG,
    *,
    log_args: bool = False,
    log_result: bool = False,
    entry: bool = True,
    exit: bool = True,
):
    """Decorator that logs function entry, exit, and exceptions.

    Args:
        level: Log level for entry/exit messages.
        log_args: Include function arguments in entry log.
        log_result: Include return value in exit log.
        entry: Log function entry.
        exit: Log function exit.
    """
    def decorator(fn):
        logger = get_logger(fn.__module__)
        fn_name = fn.__qualname__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                _kwargs: dict[str, Any] = {}
                if log_args:
                    sig = inspect.signature(fn)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    _kwargs = {k: v for k, v in bound.arguments.items() if k != "self"}
                if entry:
                    logger._log(level, f"\u2192 {fn_name}", (), _kwargs)
                t0 = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                    if exit:
                        exit_kw: dict[str, Any] = {}
                        if log_result:
                            exit_kw["result"] = result
                        ms = int((time.monotonic() - t0) * 1000)
                        if ms >= 1000:
                            exit_kw["elapsed"] = f"{ms / 1000:.1f}s"
                        else:
                            exit_kw["elapsed"] = f"{ms}ms"
                        logger._log(level, f"\u2190 {fn_name}", (), exit_kw)
                    return result
                except Exception as e:
                    ms = int((time.monotonic() - t0) * 1000)
                    logger._log(
                        logging.ERROR, f"\u2717 {fn_name}", (),
                        {"error": str(e), "elapsed": f"{ms}ms"},
                    )
                    raise
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                _kwargs: dict[str, Any] = {}
                if log_args:
                    sig = inspect.signature(fn)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    _kwargs = {k: v for k, v in bound.arguments.items() if k != "self"}
                if entry:
                    logger._log(level, f"\u2192 {fn_name}", (), _kwargs)
                t0 = time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                    if exit:
                        exit_kw: dict[str, Any] = {}
                        if log_result:
                            exit_kw["result"] = result
                        ms = int((time.monotonic() - t0) * 1000)
                        if ms >= 1000:
                            exit_kw["elapsed"] = f"{ms / 1000:.1f}s"
                        else:
                            exit_kw["elapsed"] = f"{ms}ms"
                        logger._log(level, f"\u2190 {fn_name}", (), exit_kw)
                    return result
                except Exception as e:
                    ms = int((time.monotonic() - t0) * 1000)
                    logger._log(
                        logging.ERROR, f"\u2717 {fn_name}", (),
                        {"error": str(e), "elapsed": f"{ms}ms"},
                    )
                    raise
            return sync_wrapper
    return decorator
