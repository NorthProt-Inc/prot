"""Logging constants: module map, colors."""

from __future__ import annotations

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


def module_key(name: str) -> str:
    """Extract last dotted segment: 'prot.pipeline' -> 'pipeline'."""
    return name.rsplit(".", 1)[-1]
