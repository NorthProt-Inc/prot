"""Daily JSONL conversation logger."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))


class ConversationLogger:
    """Append conversation entries to a daily JSONL file (KST-based).

    Each line is a self-contained JSON object — O(1) writes with no
    need to read or parse the existing file.
    """

    def __init__(self, log_dir: Path | str = "logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _today_file(self) -> Path:
        return self._log_dir / f"{datetime.now(KST).strftime('%Y-%m-%d')}.jsonl"

    def log(self, role: str, content: str) -> None:
        entry = {
            "timestamp": datetime.now(KST).isoformat(),
            "role": role,
            "content": content,
        }
        with open(self._today_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_today(self) -> list[dict]:
        """Read all entries from today's log file."""
        path = self._today_file()
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries
