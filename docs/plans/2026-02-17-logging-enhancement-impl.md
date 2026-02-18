# Logging Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modularize and enhance prot's logging system with async file logging, `@logged` decorator, abbreviation dictionary, and standardized multi-output format.

**Architecture:** Replace the monolithic `log.py` with a `src/prot/logging/` package containing 6 focused modules. Async file I/O via `QueueHandler` + `QueueListener` keeps hot-path latency at zero. The old `log.py` becomes a backward-compat facade so existing imports continue working.

**Tech Stack:** Python stdlib `logging`, `logging.handlers.QueueHandler/QueueListener/RotatingFileHandler`, `contextvars`, `functools.wraps`, `inspect.iscoroutinefunction`

---

### Task 1: Create constants module

**Files:**
- Create: `src/prot/logging/__init__.py` (empty placeholder for now)
- Create: `src/prot/logging/constants.py`
- Test: `tests/test_logging_constants.py`

**Step 1: Create package directory and empty `__init__.py`**

```bash
mkdir -p src/prot/logging
```

```python
# src/prot/logging/__init__.py
```

**Step 2: Write the failing test**

```python
# tests/test_logging_constants.py
"""Tests for logging constants module."""

from prot.logging.constants import (
    MODULE_MAP,
    LEVEL_COLORS,
    ABBREVIATIONS,
    RESET,
    DIM,
    abbreviate,
)


class TestModuleMap:
    def test_contains_core_modules(self):
        for mod in ("pipeline", "stt", "llm", "tts", "playback", "vad", "memory"):
            assert mod in MODULE_MAP

    def test_entries_have_abbrev_and_color(self):
        for mod, (abbrev, color) in MODULE_MAP.items():
            assert len(abbrev) <= 3
            assert color.startswith("\033[")


class TestAbbreviations:
    def test_contains_common_terms(self):
        assert ABBREVIATIONS["request"] == "req"
        assert ABBREVIATIONS["response"] == "res"
        assert ABBREVIATIONS["message"] == "msg"
        assert ABBREVIATIONS["connection"] == "conn"

    def test_abbreviate_replaces_words(self):
        assert abbreviate("request received") == "req recv"

    def test_abbreviate_preserves_unknown(self):
        assert abbreviate("hello world") == "hello world"

    def test_abbreviate_case_insensitive(self):
        result = abbreviate("Request completed")
        assert result == "req done"

    def test_abbreviate_empty_string(self):
        assert abbreviate("") == ""


class TestAnsiCodes:
    def test_reset_code(self):
        assert RESET == "\033[0m"

    def test_dim_code(self):
        assert DIM == "\033[2m"

    def test_level_colors_cover_all_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            assert level in LEVEL_COLORS
```

**Step 3: Run test to verify it fails**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_constants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prot.logging.constants'`

**Step 4: Write constants.py**

```python
# src/prot/logging/constants.py
"""Logging constants: module map, colors, abbreviations."""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# Module color / abbreviation map
# ---------------------------------------------------------------------------

MODULE_MAP: dict[str, tuple[str, str]] = {
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

RESET = "\033[0m"
DIM = "\033[2m"

LEVEL_COLORS: dict[str, str] = {
    "DEBUG":    "\033[37m",
    "INFO":     "\033[97m",
    "WARNING":  "\033[93m",
    "ERROR":    "\033[91m",
    "CRITICAL": "\033[91;1m",
}

# ---------------------------------------------------------------------------
# Abbreviation dictionary
# ---------------------------------------------------------------------------

ABBREVIATIONS: dict[str, str] = {
    # General
    "request": "req", "response": "res", "message": "msg",
    "error": "err", "config": "cfg", "connection": "conn",
    "timeout": "tout", "memory": "mem", "context": "ctx",
    "tokens": "tok", "function": "fn", "parameter": "param",
    "execution": "exec", "initialization": "init",
    "milliseconds": "ms", "seconds": "sec", "count": "cnt",
    "length": "len", "session": "sess", "entity": "ent",
    "device": "dev", "assistant": "asst",
    # Actions
    "received": "recv", "sent": "sent", "success": "ok",
    "failure": "fail", "processing": "proc", "completed": "done",
    "started": "start", "finished": "fin",
    # Technical
    "database": "db", "query": "qry", "result": "res",
    "latency": "lat", "duration": "dur",
    "provider": "prov", "model": "mdl",
}

_ABBREV_PATTERN: re.Pattern | None = None


def _get_abbrev_pattern() -> re.Pattern:
    global _ABBREV_PATTERN
    if _ABBREV_PATTERN is None:
        escaped = [re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)]
        _ABBREV_PATTERN = re.compile(
            r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE
        )
    return _ABBREV_PATTERN


def abbreviate(msg: str) -> str:
    """Replace known words with abbreviations. Disabled by NO_ABBREV=1."""
    if not msg or os.environ.get("NO_ABBREV"):
        return msg
    return _get_abbrev_pattern().sub(
        lambda m: ABBREVIATIONS[m.group(0).lower()], msg
    )


def module_key(name: str) -> str:
    """Extract last dotted segment: 'prot.pipeline' -> 'pipeline'."""
    return name.rsplit(".", 1)[-1]
```

**Step 5: Run test to verify it passes**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_constants.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/prot/logging/__init__.py src/prot/logging/constants.py tests/test_logging_constants.py
git commit -m "feat(logging): add constants module with module map, colors, abbreviations"
```

---

### Task 2: Create formatters module

**Files:**
- Create: `src/prot/logging/formatters.py`
- Test: `tests/test_logging_formatters.py`

**Step 1: Write the failing test**

```python
# tests/test_logging_formatters.py
"""Tests for logging formatters."""

import json
import logging

from prot.logging.formatters import SmartFormatter, PlainFormatter, JsonFormatter


def _make_record(msg="test message", level=logging.INFO, name="prot.pipeline", **extra):
    """Helper to create a LogRecord with extra_data."""
    record = logging.LogRecord(
        name=name, level=level, pathname="", lineno=0,
        msg=msg, args=(), exc_info=None,
    )
    record.extra_data = extra
    return record


class TestSmartFormatter:
    def test_output_contains_module_abbrev(self):
        fmt = SmartFormatter()
        record = _make_record(name="prot.pipeline")
        line = fmt.format(record)
        assert "PIP" in line
        assert "pipeline" in line

    def test_output_contains_kv_pairs(self):
        fmt = SmartFormatter()
        record = _make_record(port=8000, env="prod")
        line = fmt.format(record)
        assert "port=8000" in line
        assert "env=prod" in line

    def test_output_contains_level(self):
        fmt = SmartFormatter()
        record = _make_record(level=logging.WARNING)
        line = fmt.format(record)
        assert "WARNI" in line or "WARNING" in line


class TestPlainFormatter:
    def test_no_ansi_codes(self):
        fmt = PlainFormatter()
        record = _make_record(name="prot.stt")
        line = fmt.format(record)
        assert "\033[" not in line

    def test_contains_full_date(self):
        fmt = PlainFormatter()
        record = _make_record()
        line = fmt.format(record)
        # YYYY-MM-DD format
        assert "-" in line.split(" ")[0]

    def test_contains_kv_pairs(self):
        fmt = PlainFormatter()
        record = _make_record(attempt=3)
        line = fmt.format(record)
        assert "attempt=3" in line


class TestJsonFormatter:
    def test_output_is_valid_json(self):
        fmt = JsonFormatter()
        record = _make_record(port=8000)
        line = fmt.format(record)
        data = json.loads(line)
        assert data["msg"] == "test message"

    def test_includes_extra_data(self):
        fmt = JsonFormatter()
        record = _make_record(port=8000, env="prod")
        data = json.loads(fmt.format(record))
        assert data["port"] == 8000
        assert data["env"] == "prod"

    def test_includes_level_and_logger(self):
        fmt = JsonFormatter()
        record = _make_record(name="prot.tts", level=logging.ERROR)
        data = json.loads(fmt.format(record))
        assert data["level"] == "ERROR"
        assert data["logger"] == "prot.tts"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_formatters.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write formatters.py**

```python
# src/prot/logging/formatters.py
"""Log formatters: colored console, plain file, JSON."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from prot.logging.constants import (
    MODULE_MAP, LEVEL_COLORS, RESET, DIM, abbreviate, module_key,
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

        # Append elapsed if available
        elapsed = getattr(record, "elapsed_ms", None)
        if elapsed is not None:
            if elapsed >= 1000:
                kv_parts.append(f"+{elapsed / 1000:.1f}s")
            else:
                kv_parts.append(f"+{elapsed}ms")

        kv_str = f" {DIM}| {' '.join(kv_parts)}{RESET}" if kv_parts else ""
        msg = abbreviate(record.getMessage())

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
        msg = abbreviate(record.getMessage())

        line = f"{timestamp} {record.levelname:<8} [{mod}] {msg}{kv_str}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line += "\n" + record.exc_text
        return line


class JsonFormatter(logging.Formatter):
    """JSONL formatter: one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        data: dict = {
            "ts": dt.isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra_data: dict = getattr(record, "extra_data", {})
        data.update(extra_data)

        elapsed = getattr(record, "elapsed_ms", None)
        if elapsed is not None:
            data["elapsed_ms"] = elapsed

        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_formatters.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/prot/logging/formatters.py tests/test_logging_formatters.py
git commit -m "feat(logging): add formatters — SmartFormatter, PlainFormatter, JsonFormatter"
```

---

### Task 3: Create structured_logger module (StructuredLogger + turn tracking)

**Files:**
- Create: `src/prot/logging/structured_logger.py`
- Test: `tests/test_logging_structured.py`

**Step 1: Write the failing test**

```python
# tests/test_logging_structured.py
"""Tests for StructuredLogger and turn tracking."""

import logging

from prot.logging.structured_logger import (
    StructuredLogger,
    get_logger,
    start_turn,
    elapsed_ms,
    reset_turn,
)


class TestStructuredLogger:
    def test_info_logs_message(self, caplog):
        with caplog.at_level(logging.INFO, logger="test.structured"):
            sl = StructuredLogger(logging.getLogger("test.structured"))
            sl.info("hello")
        assert "hello" in caplog.text

    def test_kwargs_stored_as_extra_data(self):
        inner = logging.getLogger("test.extra")
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)

        sl = StructuredLogger(inner)
        sl.info("msg", port=8000, env="prod")

        assert len(records) == 1
        assert records[0].extra_data == {"port": 8000, "env": "prod"}
        inner.removeHandler(handler)

    def test_exception_includes_exc_info(self):
        inner = logging.getLogger("test.exc")
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)

        sl = StructuredLogger(inner)
        try:
            raise ValueError("boom")
        except ValueError:
            sl.exception("failed")

        assert records[0].exc_info is not None
        inner.removeHandler(handler)


class TestGetLogger:
    def test_returns_structured_logger(self):
        logger = get_logger("test.factory")
        assert isinstance(logger, StructuredLogger)

    def test_same_name_returns_same_instance(self):
        a = get_logger("test.same")
        b = get_logger("test.same")
        assert a is b


class TestTurnTracking:
    def test_elapsed_none_before_start(self):
        reset_turn()
        assert elapsed_ms() is None

    def test_elapsed_returns_int_after_start(self):
        start_turn()
        ms = elapsed_ms()
        assert isinstance(ms, int)
        assert ms >= 0
        reset_turn()

    def test_reset_clears_timer(self):
        start_turn()
        reset_turn()
        assert elapsed_ms() is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_structured.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write structured_logger.py**

```python
# src/prot/logging/structured_logger.py
"""StructuredLogger wrapper and turn tracking."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar

from prot.logging.constants import abbreviate

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
        extra_data = kwargs
        record = self._logger.makeRecord(
            self._logger.name, level, "", 0, msg, args,
            exc_info=exc_info, extra={"extra_data": extra_data},
        )
        record.extra_data = extra_data  # ensure direct attribute access
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
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_structured.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/prot/logging/structured_logger.py tests/test_logging_structured.py
git commit -m "feat(logging): add StructuredLogger with turn tracking"
```

---

### Task 4: Create handlers module (async file I/O)

**Files:**
- Create: `src/prot/logging/handlers.py`
- Test: `tests/test_logging_handlers.py`

**Step 1: Write the failing test**

```python
# tests/test_logging_handlers.py
"""Tests for async file logging handlers."""

import logging
import time
from pathlib import Path

from prot.logging.handlers import create_async_handler, create_error_handler


class TestAsyncHandler:
    def test_creates_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler, listener = create_async_handler(str(log_file))
        listener.start()

        logger = logging.getLogger("test.async_handler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("hello async")

        # Give listener time to flush
        time.sleep(0.1)
        listener.stop()
        logger.removeHandler(handler)

        assert log_file.exists()
        content = log_file.read_text()
        assert "hello async" in content

    def test_rotates_on_max_bytes(self, tmp_path):
        log_file = tmp_path / "rotate.log"
        handler, listener = create_async_handler(
            str(log_file), max_bytes=100, backup_count=2
        )
        listener.start()

        logger = logging.getLogger("test.rotate")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        for i in range(50):
            logger.info(f"line {i} " + "x" * 50)

        time.sleep(0.2)
        listener.stop()
        logger.removeHandler(handler)

        # Should have rotated files
        log_files = list(tmp_path.glob("rotate.log*"))
        assert len(log_files) > 1


class TestErrorHandler:
    def test_only_captures_errors(self, tmp_path):
        log_file = tmp_path / "error.log"
        handler, listener = create_error_handler(str(log_file))
        listener.start()

        logger = logging.getLogger("test.error_filter")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)
        listener.stop()
        logger.removeHandler(handler)

        content = log_file.read_text()
        assert "info message" not in content
        assert "warning message" not in content
        assert "error message" in content
        assert "critical message" in content
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_handlers.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write handlers.py**

```python
# src/prot/logging/handlers.py
"""Async file logging handlers using QueueHandler + QueueListener."""

from __future__ import annotations

import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from queue import Queue


def create_async_handler(
    log_path: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    formatter: logging.Formatter | None = None,
) -> tuple[QueueHandler, QueueListener]:
    """Create an async file handler.

    Returns (queue_handler, listener). Caller must call listener.start()
    and listener.stop() at shutdown.
    """
    queue: Queue = Queue(-1)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    if formatter:
        file_handler.setFormatter(formatter)
    else:
        file_handler.setFormatter(logging.Formatter("%(message)s"))
    listener = QueueListener(queue, file_handler, respect_handler_level=True)
    queue_handler = QueueHandler(queue)
    return queue_handler, listener


def create_error_handler(
    log_path: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    formatter: logging.Formatter | None = None,
) -> tuple[QueueHandler, QueueListener]:
    """Create an async error-only file handler (ERROR and CRITICAL)."""
    queue: Queue = Queue(-1)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    file_handler.setLevel(logging.ERROR)
    if formatter:
        file_handler.setFormatter(formatter)
    else:
        file_handler.setFormatter(logging.Formatter("%(message)s"))
    listener = QueueListener(queue, file_handler, respect_handler_level=True)
    queue_handler = QueueHandler(queue)
    return queue_handler, listener
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_handlers.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/prot/logging/handlers.py tests/test_logging_handlers.py
git commit -m "feat(logging): add async QueueHandler-based file handlers"
```

---

### Task 5: Create decorator module (@logged)

**Files:**
- Create: `src/prot/logging/decorator.py`
- Test: `tests/test_logging_decorator.py`

**Step 1: Write the failing test**

```python
# tests/test_logging_decorator.py
"""Tests for @logged decorator."""

import logging

import pytest

from prot.logging.decorator import logged
from prot.logging.structured_logger import get_logger


class TestLoggedDecorator:
    def test_sync_function_entry_exit(self, caplog):
        @logged()
        def add(a, b):
            return a + b

        with caplog.at_level(logging.DEBUG):
            result = add(1, 2)
        assert result == 3
        messages = caplog.text
        assert "\u2192" in messages or "→" in messages  # entry arrow
        assert "\u2190" in messages or "←" in messages  # exit arrow

    @pytest.mark.asyncio
    async def test_async_function_entry_exit(self, caplog):
        @logged()
        async def fetch(url):
            return "data"

        with caplog.at_level(logging.DEBUG):
            result = await fetch("http://example.com")
        assert result == "data"
        messages = caplog.text
        assert "\u2192" in messages or "→" in messages

    def test_exception_logged_with_marker(self, caplog):
        @logged()
        def fail():
            raise ValueError("boom")

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError, match="boom"):
                fail()
        messages = caplog.text
        assert "\u2717" in messages or "✗" in messages  # error marker

    def test_log_args_includes_arguments(self, caplog):
        @logged(log_args=True)
        def greet(name):
            return f"hi {name}"

        with caplog.at_level(logging.DEBUG):
            greet("alice")
        assert "name=alice" in caplog.text

    def test_log_result_includes_return_value(self, caplog):
        @logged(log_result=True)
        def double(x):
            return x * 2

        with caplog.at_level(logging.DEBUG):
            double(5)
        assert "result=10" in caplog.text

    def test_custom_level(self, caplog):
        @logged(level=logging.WARNING)
        def important():
            return 42

        with caplog.at_level(logging.WARNING):
            important()
        assert "important" in caplog.text
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_decorator.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write decorator.py**

```python
# src/prot/logging/decorator.py
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
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_decorator.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/prot/logging/decorator.py tests/test_logging_decorator.py
git commit -m "feat(logging): add @logged decorator for auto entry/exit logging"
```

---

### Task 6: Create setup module + wire __init__.py

**Files:**
- Create: `src/prot/logging/setup.py`
- Modify: `src/prot/logging/__init__.py`
- Test: `tests/test_logging_setup.py`

**Step 1: Write the failing test**

```python
# tests/test_logging_setup.py
"""Tests for logging setup and public API."""

import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

from prot.logging import (
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
from prot.logging.setup import _listeners


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "testlogs"
        setup_logging(level="DEBUG", log_dir=str(log_dir))
        assert log_dir.exists()

    def test_creates_prot_log_file(self, tmp_path):
        log_dir = tmp_path / "testlogs2"
        setup_logging(level="DEBUG", log_dir=str(log_dir))
        logger = get_logger("test.setup")
        logger.info("setup test")
        time.sleep(0.2)
        assert (log_dir / "prot.log").exists()

    def test_creates_error_log_file(self, tmp_path):
        log_dir = tmp_path / "testlogs3"
        setup_logging(level="DEBUG", log_dir=str(log_dir))
        logger = get_logger("test.error_setup")
        logger.error("error test")
        time.sleep(0.2)
        assert (log_dir / "prot_error.log").exists()

    def test_json_logging_opt_in(self, tmp_path):
        log_dir = tmp_path / "testlogs4"
        with patch.dict(os.environ, {"LOG_JSON": "true"}):
            setup_logging(level="DEBUG", log_dir=str(log_dir))
        logger = get_logger("test.json")
        logger.info("json test")
        time.sleep(0.2)
        assert (log_dir / "prot.jsonl").exists()


class TestPublicAPI:
    def test_get_logger_accessible(self):
        logger = get_logger("test.api")
        assert isinstance(logger, StructuredLogger)

    def test_turn_tracking_accessible(self):
        start_turn()
        ms = elapsed_ms()
        assert ms is not None
        reset_turn()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_setup.py -v`
Expected: FAIL

**Step 3: Write setup.py**

```python
# src/prot/logging/setup.py
"""Logging setup: configure root logger with console + async file handlers."""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path

from prot.logging.formatters import SmartFormatter, PlainFormatter, JsonFormatter
from prot.logging.handlers import create_async_handler, create_error_handler

_listeners: list = []


def setup_logging(
    level: str | None = None,
    log_dir: str = "logs",
) -> None:
    """Configure root logger with console + async file handlers.

    Level resolution: explicit arg > LOG_LEVEL env > config default.
    """
    from prot.config import settings

    resolved = level or os.environ.get("LOG_LEVEL") or settings.log_level
    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    # Stop any existing listeners
    for listener in _listeners:
        listener.stop()
    _listeners.clear()

    # 1. Console handler (colored)
    console = logging.StreamHandler()
    console.setFormatter(SmartFormatter())
    root.addHandler(console)

    # 2. File handlers (async)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 2a. Main log file (rotated, plain text)
    file_handler, file_listener = create_async_handler(
        str(log_path / "prot.log"),
        formatter=PlainFormatter(),
    )
    root.addHandler(file_handler)
    file_listener.start()
    _listeners.append(file_listener)

    # 2b. Error-only log file
    error_handler, error_listener = create_error_handler(
        str(log_path / "prot_error.log"),
        formatter=PlainFormatter(),
    )
    root.addHandler(error_handler)
    error_listener.start()
    _listeners.append(error_listener)

    # 3. Optional JSONL handler
    json_enabled = os.environ.get("LOG_JSON", "").lower() in ("1", "true", "yes")
    if json_enabled:
        json_handler, json_listener = create_async_handler(
            str(log_path / "prot.jsonl"),
            formatter=JsonFormatter(),
        )
        root.addHandler(json_handler)
        json_listener.start()
        _listeners.append(json_listener)

    # Ensure listeners stop on exit
    atexit.register(_shutdown_listeners)


def _shutdown_listeners() -> None:
    for listener in _listeners:
        try:
            listener.stop()
        except Exception:
            pass
    _listeners.clear()
```

**Step 4: Write `__init__.py` with public API**

```python
# src/prot/logging/__init__.py
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
```

**Step 5: Run test to verify it passes**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_logging_setup.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/prot/logging/setup.py src/prot/logging/__init__.py tests/test_logging_setup.py
git commit -m "feat(logging): add setup_logging with async file handlers and public API"
```

---

### Task 7: Migrate log.py to backward-compat facade + delete legacy files

**Files:**
- Modify: `src/prot/log.py` — replace with facade
- Delete: `src/prot/logger.py`
- Delete: `tests/test_logger.py`
- Test: `tests/test_logging_compat.py`

**Step 1: Write the failing test**

```python
# tests/test_logging_compat.py
"""Tests for backward compatibility via prot.log facade."""

from prot.log import (
    get_logger,
    setup_logging,
    start_turn,
    elapsed_ms,
    reset_turn,
    StructuredLogger,
    SmartFormatter,
)
from prot.logging import get_logger as new_get_logger


class TestBackwardCompat:
    def test_get_logger_works(self):
        logger = get_logger("test.compat")
        assert isinstance(logger, StructuredLogger)

    def test_setup_logging_callable(self):
        assert callable(setup_logging)

    def test_turn_tracking_works(self):
        start_turn()
        ms = elapsed_ms()
        assert ms is not None
        reset_turn()
        assert elapsed_ms() is None

    def test_formatter_importable(self):
        assert SmartFormatter is not None
```

**Step 2: Run existing tests to capture baseline**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/ -v --ignore=tests/test_logger.py -k "not integration" 2>&1 | tail -20`

**Step 3: Replace log.py with facade**

```python
# src/prot/log.py
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
```

**Step 4: Delete legacy files**

```bash
rm src/prot/logger.py tests/test_logger.py
```

**Step 5: Run all tests**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/ -v -k "not integration"`
Expected: All PASS — existing imports via `prot.log` still work

**Step 6: Commit**

```bash
git add src/prot/log.py tests/test_logging_compat.py
git rm src/prot/logger.py tests/test_logger.py
git commit -m "refactor(logging): replace log.py with facade, remove legacy logger.py"
```

---

### Task 8: Update .gitignore and .env.example

**Files:**
- Modify: `.gitignore` — add `logs/`
- Modify: `.env.example` — add new env vars

**Step 1: Add logs/ to .gitignore**

Append to `.gitignore`:
```
# Application logs
logs/
```

**Step 2: Update .env.example**

Replace the logging section in `.env.example`:
```
# ─── Logging ─────────────────────────────────────────────────
LOG_LEVEL=INFO
# LOG_JSON=false
# LOG_DIR=logs
# NO_COLOR=
# NO_ABBREV=
```

**Step 3: Run all tests to verify nothing broke**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/ -v -k "not integration"`
Expected: All PASS

**Step 4: Commit**

```bash
git add .gitignore .env.example
git commit -m "chore: add logs/ to gitignore, update env example with logging vars"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run full test suite**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/ -v -k "not integration"`
Expected: All PASS

**Step 2: Verify log output manually**

```bash
cd /home/cyan/workplace/prot && python -c "
from prot.logging import setup_logging, get_logger, logged, start_turn, reset_turn
import time

setup_logging(level='DEBUG', log_dir='/tmp/prot_test_logs')
log = get_logger('test')

start_turn()
log.info('Server started', port=8000, env='dev')
log.debug('Processing request', user_id='abc123')
log.warning('Rate limit approaching', remaining=10)

@logged(log_args=True, log_result=True)
def add(a, b):
    return a + b

add(3, 4)

time.sleep(0.1)
log.error('Connection failed', host='db.local', retry=3)
reset_turn()

print()
print('--- /tmp/prot_test_logs/prot.log ---')
print(open('/tmp/prot_test_logs/prot.log').read())
print('--- /tmp/prot_test_logs/prot_error.log ---')
print(open('/tmp/prot_test_logs/prot_error.log').read())
"
```

Expected: Console shows colored output, file shows plain output, error file only has error line.

**Step 3: Final commit with all tests passing**

If any adjustments were needed, commit them:
```bash
git add -A && git commit -m "fix: final adjustments from integration testing"
```
