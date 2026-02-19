"""Daily JSONL conversation log archival."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

from prot.logging import get_logger
from prot.processing import content_to_text

logger = get_logger(__name__)

LOCAL_TZ = ZoneInfo("America/Vancouver")


class ConversationLogger:
    """Save conversation sessions as daily JSONL files."""

    def __init__(self, log_dir: str = "data/conversations") -> None:
        self._log_dir = Path(log_dir)

    def save_session(
        self, session_id: UUID, messages: list[dict]
    ) -> Path | None:
        """Append session to daily JSONL file. Returns path or None on error."""
        if not messages:
            return None
        try:
            now = datetime.now(LOCAL_TZ)
            today = now.strftime("%Y-%m-%d")
            path = self._log_dir / f"{today}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)

            serializable = []
            for m in messages:
                serializable.append({
                    "role": m["role"],
                    "content": content_to_text(m.get("content", "")),
                })

            record = json.dumps({
                "session_id": str(session_id),
                "timestamp": now.isoformat(),
                "messages": serializable,
            }, ensure_ascii=False)

            with path.open("a", encoding="utf-8") as f:
                f.write(record + "\n")

            logger.info("Session saved", path=str(path), messages=len(messages))
            return path
        except Exception:
            logger.exception("Failed to save session")
            return None
