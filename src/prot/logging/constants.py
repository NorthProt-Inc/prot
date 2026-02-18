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
