"""Log formatters: colored console, plain file."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from prot.logging.constants import (
    MODULE_MAP, LEVEL_COLORS, RESET, DIM, module_key,
)


class SmartFormatter(logging.Formatter):
    """Console formatter: colored output with k=v pairs and elapsed time."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        ms = int(record.created * 1000) % 1000
        timestamp = f"{ts}.{ms:03d}"

        mod = module_key(record.name)
        abbrev, color = MODULE_MAP.get(mod, (mod[:3].upper(), "\033[37m"))
        level_color = LEVEL_COLORS.get(record.levelname, "")

        extra_data: dict = getattr(record, "extra_data", {})
        kv_parts = [f"{k}={v}" for k, v in extra_data.items()]

        elapsed = getattr(record, "elapsed_ms", None)
        if elapsed is not None:
            if elapsed >= 1000:
                kv_parts.append(f"+{elapsed / 1000:.1f}s")
            else:
                kv_parts.append(f"+{elapsed}ms")

        kv_str = f" {DIM}| {' '.join(kv_parts)}{RESET}" if kv_parts else ""
        msg = record.getMessage()

        line = (
            f"{DIM}{timestamp}{RESET}  "
            f"{level_color}{record.levelname:<5}{RESET} "
            f"[{color}{abbrev}{RESET}|{color}{mod:<10}{RESET}] "
            f"{msg}{kv_str}"
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
        extra_data: dict = getattr(record, "extra_data", {})
        kv_parts = [f"{k}={v}" for k, v in extra_data.items()]

        elapsed = getattr(record, "elapsed_ms", None)
        if elapsed is not None:
            if elapsed >= 1000:
                kv_parts.append(f"+{elapsed / 1000:.1f}s")
            else:
                kv_parts.append(f"+{elapsed}ms")

        kv_str = f" | {' '.join(kv_parts)}" if kv_parts else ""
        msg = record.getMessage()

        line = f"{timestamp} {record.levelname:<8} [{mod}] {msg}{kv_str}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line += "\n" + record.exc_text
        return line
