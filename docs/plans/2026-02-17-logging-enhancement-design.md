# Logging Enhancement Design

**Date:** 2026-02-17
**Status:** Approved

---

## Overview

Enhance the prot voice pipeline logging system with:
- Modularized package structure (`src/prot/logging/`)
- Async file logging (QueueHandler + worker thread) — zero hot-path I/O
- `@logged` decorator for automatic function entry/exit logging
- Abbreviation dictionary for concise log messages
- Standardized log output: console (colored), file (plain, rotated), error-only file, optional JSONL
- `logs/` directory at project root for system log access
- Remove legacy `logger.py` (replaced by `conversation_log.py`)

---

## Package Structure

```
src/prot/logging/
├── __init__.py            # Public API exports
├── constants.py           # MODULE_MAP, LEVEL_COLORS, ABBREVIATIONS, ANSI codes
├── formatters.py          # SmartFormatter (console), PlainFormatter (file), JsonFormatter (JSONL)
├── structured_logger.py   # StructuredLogger class + get_logger() + turn tracking
├── decorator.py           # @logged() decorator (async/sync, entry/exit logging)
├── handlers.py            # AsyncQueueHandler + worker thread for file I/O
└── setup.py               # setup_logging() — root logger configuration

logs/                      # Auto-created at project root (.gitignored)
├── prot.log               # RotatingFileHandler (10MB × 5 backups, plain text, no color)
└── prot_error.log         # Errors & Critical only

src/prot/log.py            # Backward compat: re-exports from prot.logging
```

---

## Async File Logging (handlers.py)

Uses `QueueHandler` → `Queue` → `QueueListener` → `RotatingFileHandler`:

1. Log call → `QueueHandler` enqueues record (non-blocking)
2. `QueueListener` background thread dequeues and writes to file
3. Zero file I/O on hot path

Error-only handler: separate `RotatingFileHandler` filtering ERROR/CRITICAL.

---

## Formatters (formatters.py)

| Formatter | Target | Format |
|-----------|--------|--------|
| `SmartFormatter` | Console | `HH:MM:SS.mmm LEVEL [ABR\|module] msg \| k=v +elapsed` (colored) |
| `PlainFormatter` | File | `YYYY-MM-DD HH:MM:SS.mmm LEVEL [module] msg \| k=v +elapsed` (no color) |
| `JsonFormatter` | JSONL | `{"ts":..., "level":..., "logger":..., "msg":..., ...kwargs}` |

---

## @logged Decorator (decorator.py)

```python
@logged()                                    # Default: DEBUG level entry/exit
@logged(level=logging.INFO, log_args=True)   # Include arguments
@logged(log_result=True, exit=True)          # Include result, exit log

# Output:
# 14:32:01.234 DEBUG [STT|stt] → process_audio
# 14:32:01.567 DEBUG [STT|stt] ← process_audio | +333ms
# 14:32:01.567 ERROR [STT|stt] ✗ process_audio | error=ConnectionError
```

Features:
- Auto-detects `async def` / `def`
- Exception logging with `✗` marker
- Elapsed time measurement per call
- `log_args=True` to include argument values

---

## Abbreviation Dictionary (constants.py)

Applied to log messages only (not k=v keys). Disabled via `NO_ABBREV=1`.

```python
ABBREVIATIONS = {
    "request": "req", "response": "res", "message": "msg",
    "error": "err", "connection": "conn", "timeout": "tout",
    "memory": "mem", "context": "ctx", "tokens": "tok",
    "function": "fn", "parameter": "param", "execution": "exec",
    "milliseconds": "ms", "seconds": "sec", "session": "sess",
    "received": "recv", "completed": "done", "started": "start",
    "processing": "proc", "latency": "lat", "duration": "dur",
    "provider": "prov", "model": "mdl",
}
```

---

## setup_logging() (setup.py)

```python
def setup_logging(level=None, log_dir="logs", json_logging=False):
    """
    1. Console handler (SmartFormatter, colored)
    2. File handler (PlainFormatter, rotated 10MB × 5) — async via QueueHandler
    3. Error file handler (ERROR+ only) — async
    4. Optional: JSONL handler (JsonFormatter) — LOG_JSON=true
    5. logs/ directory auto-created
    """
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_JSON` | `false` | Enable JSONL logging |
| `LOG_DIR` | `logs` | Log directory path |
| `NO_COLOR` | unset | Disable console colors |
| `NO_ABBREV` | unset | Disable abbreviations |

---

## Migration Strategy

- `src/prot/log.py` becomes backward-compat facade (re-exports from `prot.logging`)
- All existing `from prot.log import ...` continue to work
- New code uses `from prot.logging import ...`
- Delete `src/prot/logger.py` + `tests/test_logger.py`
- Update all imports in codebase to new paths (optional, can be gradual)

---

## Files to Create/Modify/Delete

**Create:**
- `src/prot/logging/__init__.py`
- `src/prot/logging/constants.py`
- `src/prot/logging/formatters.py`
- `src/prot/logging/structured_logger.py`
- `src/prot/logging/decorator.py`
- `src/prot/logging/handlers.py`
- `src/prot/logging/setup.py`

**Modify:**
- `src/prot/log.py` → backward compat facade
- `.gitignore` → add `logs/`
- `.env.example` → add new env vars

**Delete:**
- `src/prot/logger.py`
- `tests/test_logger.py`
