"""Daily JSON conversation log archival."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

from prot.log import get_logger

logger = get_logger(__name__)

LOCAL_TZ = ZoneInfo("America/Vancouver")


def _content_to_text(content) -> str:
    """Extract plain text from str or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.text if hasattr(block, "text") else
            str(block.get("text", "")) if isinstance(block, dict) else ""
            for block in content
        )
    return str(content)


class ConversationLogger:
    """Save conversation sessions as daily JSON files."""

    def __init__(self, log_dir: str = "data/conversations") -> None:
        self._log_dir = Path(log_dir)

    def save_session(
        self, session_id: UUID, messages: list[dict]
    ) -> Path | None:
        """Write conversation to JSON file. Returns path or None on error."""
        if not messages:
            return None
        try:
            now = datetime.now(LOCAL_TZ)
            today = now.strftime("%Y-%m-%d")
            path = self._log_dir / f"{today}-{str(session_id)[:8]}.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            serializable = []
            for m in messages:
                serializable.append({
                    "role": m["role"],
                    "content": _content_to_text(m.get("content", "")),
                })

            data = {
                "session_id": str(session_id),
                "timestamp": now.isoformat(),
                "messages": serializable,
                "version": "1.0",
            }
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Session saved", path=str(path), messages=len(messages))
            return path
        except Exception:
            logger.exception("Failed to save session")
            return None
