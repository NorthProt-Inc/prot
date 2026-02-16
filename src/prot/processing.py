import re

_RE_MARKDOWN = re.compile(r'[*_#`~\[\](){}|>]')
_RE_NUMBERED = re.compile(r'\d+\.\s')
_RE_BULLETS = re.compile(r'[-•]\s')
_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?~])\s+')
_RE_SENTENCE_END = re.compile(r'[.!?~]$')


def sanitize_for_tts(text: str) -> str:
    text = _RE_MARKDOWN.sub('', text)
    text = _RE_NUMBERED.sub('', text)
    text = _RE_BULLETS.sub('', text)
    return text.strip()


def ensure_complete_sentence(text: str) -> str:
    for i in range(len(text) - 1, -1, -1):
        if text[i] in '.!?~':
            return text[:i + 1]
    return text


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
    return complete, remainder
