"""Daily JSON conversation logger."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))


class ConversationLogger:
    """Append conversation entries to a daily JSON file (KST-based)."""

    def __init__(self, log_dir: Path | str = "logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _today_file(self) -> Path:
        return self._log_dir / f"{datetime.now(KST).strftime('%Y-%m-%d')}.json"

    def log(self, role: str, content: str) -> None:
        path = self._today_file()
        entries: list[dict] = []
        if path.exists():
            entries = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "timestamp": datetime.now(KST).isoformat(),
                "role": role,
                "content": content,
            }
        )
        path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
