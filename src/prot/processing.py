from __future__ import annotations

import re

_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?~])\s+')
_RE_SENTENCE_END = re.compile(r'[.!?~]$')

MAX_BUFFER_CHARS = 2000

_RE_SPACE_BEFORE_BRACKET = re.compile(r'(\w)\[')
_RE_SPACE_AFTER_BRACKET = re.compile(r'\](\w)')


def sanitize_for_tts(text: str) -> str:
    """Ensure spacing around audio tag brackets for ElevenLabs v3.

    Injects space before [ when preceded by a word char,
    and after ] when followed by a word char.
    Preserves ]., ]!, ]? (no space before punctuation).
    """
    if not text:
        return text
    text = _RE_SPACE_BEFORE_BRACKET.sub(r'\1 [', text)
    text = _RE_SPACE_AFTER_BRACKET.sub(r'] \1', text)
    return text


def is_tool_result_message(msg: dict) -> bool:
    """Check if a message contains only tool_result blocks."""
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


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


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses."""
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text
