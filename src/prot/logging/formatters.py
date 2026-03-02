"""Log formatters: colored console, plain file."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from prot.logging.constants import (
    MODULE_MAP, LEVEL_COLORS, RESET, DIM, module_key,
)


def _prepare_record(record: logging.LogRecord) -> tuple[list[str], str]:
    """Extract trace metadata from record and build shared kv_parts + indent.

    Reads ``_depth`` and ``_elapsed`` from ``extra_data`` without mutating it,
    and filters ``_``-prefixed keys from k=v output.  Returns ``(kv_parts,
    indent)`` ready for both colored and plain line assembly.
    """
    extra_data: dict = getattr(record, "extra_data", {})

    # Extract trace metadata (non-mutating to support multiple formatters)
    trace_depth = extra_data.get("_depth", None)
    trace_elapsed = extra_data.get("_elapsed", None)

    kv_parts = [f"{k}={v}" for k, v in extra_data.items() if not k.startswith("_")]

    if trace_elapsed:
        kv_parts.append(trace_elapsed)
    else:
        elapsed = getattr(record, "elapsed_ms", None)
        if elapsed is not None:
            if elapsed >= 1000:
                kv_parts.append(f"+{elapsed / 1000:.1f}s")
            else:
                kv_parts.append(f"+{elapsed}ms")

    # Call-depth indent for trace messages
    indent = f"{'| ' * trace_depth}" if trace_depth is not None else ""

    return kv_parts, indent


class SmartFormatter(logging.Formatter):
    """Console formatter: colored output with k=v pairs and elapsed time."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        ms = int(record.created * 1000) % 1000
        timestamp = f"{ts}.{ms:03d}"

        mod = module_key(record.name)
        abbrev, color = MODULE_MAP.get(mod, (mod[:3].upper(), "\033[37m"))
        level_color = LEVEL_COLORS.get(record.levelname, "")

        kv_parts, indent = _prepare_record(record)
        msg = record.getMessage()

        # Severity-aware body coloring: WARNING+ gets level_color on message area
        if record.levelno >= logging.WARNING:
            kv_tail = f" | {' '.join(kv_parts)}" if kv_parts else ""
            body = f"{level_color}{indent}{msg}{kv_tail}{RESET}"
        else:
            kv_tail = f" {DIM}| {' '.join(kv_parts)}{RESET}" if kv_parts else ""
            body = f"{indent}{msg}{kv_tail}"

        line = (
            f"{DIM}{timestamp}{RESET}  "
            f"{level_color}{record.levelname:<5}{RESET} "
            f"[{color}{abbrev}{RESET}|{color}{mod:<10}{RESET}] "
            f"{body}"
        )

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line += "\n" + record.exc_text
        return line


class PlainFormatter(logging.Formatter):
    """File formatter: no ANSI codes, full date, k=v pairs."""

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(record.created * 1000) % 1000:03d}"

        mod = module_key(record.name)

        kv_parts, indent = _prepare_record(record)
        kv_str = f" | {' '.join(kv_parts)}" if kv_parts else ""
        msg = record.getMessage()

        line = f"{timestamp} {record.levelname:<8} [{mod}] {indent}{msg}{kv_str}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line += "\n" + record.exc_text
        return line
