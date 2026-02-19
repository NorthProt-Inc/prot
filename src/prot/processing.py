from __future__ import annotations

import re

_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?~])\s+')
_RE_SENTENCE_END = re.compile(r'[.!?~]$')
_RE_SPECIAL = re.compile(r'[^\w\s.,!?\[\]]')
_RE_MULTI_SPACE = re.compile(r' {2,}')

MAX_BUFFER_CHARS = 2000


def sanitize_for_tts(text: str) -> str:
    """Replace special characters with spaces to prevent TTS gluing.

    ElevenLabs silently strips special characters (*, ~, #, etc.),
    which causes adjacent text to merge without gaps.
    Preserves word chars, whitespace, punctuation (.,!?) and
    audio tag brackets ([]).
    """
    text = _RE_SPECIAL.sub(' ', text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    return text.strip()


def content_to_text(content) -> str:
    """Extract plain text from str or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.text if hasattr(block, "text") else
            str(block.get("text", "") or block.get("content", "")) if isinstance(block, dict) else ""
            for block in content
        )
    return str(content)


def chunk_sentences(text: str) -> tuple[list[str], str]:
    """Split text into complete sentences and a trailing remainder.

    Returns (complete_sentences, remainder) where remainder is the
    trailing text that does not end with a sentence terminator.
    """
    stripped = text.strip()
    if not stripped:
        return [], ""
    parts = _RE_SENTENCE_SPLIT.split(stripped)
    if not parts:
        return [], stripped
    if _RE_SENTENCE_END.search(parts[-1]):
        return [p.strip() for p in parts if p.strip()], ""
    remainder = parts.pop().strip()
    complete = [p.strip() for p in parts if p.strip()]
    if len(remainder) > MAX_BUFFER_CHARS:
        complete.append(remainder)
        remainder = ""
    return complete, remainder
