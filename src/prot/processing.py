import re

_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?~])\s+')
_RE_SENTENCE_END = re.compile(r'[.!?~]$')

MAX_BUFFER_CHARS = 2000



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
