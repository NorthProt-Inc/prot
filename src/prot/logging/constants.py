"""Logging constants: module map, colors."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module color / abbreviation map
# ---------------------------------------------------------------------------

MODULE_MAP: dict[str, tuple[str, str]] = {
    "pipeline":         ("PIP", "\033[96m"),
    "stt":              ("STT", "\033[94m"),
    "llm":              ("LLM", "\033[92m"),
    "tts":              ("TTS", "\033[38;5;208m"),
    "playback":         ("PLY", "\033[93m"),
    "vad":              ("VAD", "\033[91m"),
    "memory":           ("MEM", "\033[95m"),
    "graphrag":         ("RAG", "\033[38;5;39m"),
    "embeddings":       ("EMB", "\033[38;5;147m"),
    "db":               ("DB",  "\033[90m"),
    "context":          ("CTX", "\033[38;5;117m"),
    "audio":            ("AUD", "\033[97m"),
    "app":              ("APP", "\033[38;5;248m"),
    "config":           ("CFG", "\033[38;5;243m"),
    "hass":             ("HAS", "\033[38;5;214m"),
    "decay":            ("DCY", "\033[38;5;183m"),
    "state":            ("STA", "\033[96m"),
}

RESET = "\033[0m"
DIM = "\033[2m"

LEVEL_COLORS: dict[str, str] = {
    "DEBUG":    "\033[90m",
    "INFO":     "\033[0m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[97;41m",
}


def module_key(name: str) -> str:
    """Extract last dotted segment: 'prot.pipeline' -> 'pipeline'."""
    return name.rsplit(".", 1)[-1]
