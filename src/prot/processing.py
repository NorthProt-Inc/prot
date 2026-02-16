import re
from typing import Generator


def sanitize_for_tts(text: str) -> str:
    text = re.sub(r'[*_#`~\[\](){}|>]', '', text)  # markdown
    text = re.sub(r'\d+\.\s', '', text)              # numbered lists
    text = re.sub(r'[-•]\s', '', text)                # bullets
    return text.strip()


def ensure_complete_sentence(text: str) -> str:
    for i in range(len(text) - 1, -1, -1):
        if text[i] in '.!?~':
            return text[:i + 1]
    return text


def chunk_sentences(text: str) -> Generator[str, None, None]:
    if not text.strip():
        return
    parts = re.split(r'(?<=[.!?~])\s+', text.strip())
    for part in parts:
        part = part.strip()
        if part:
            yield part
