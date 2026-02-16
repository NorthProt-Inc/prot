import json
from pathlib import Path

import pytest

from prot.logger import ConversationLogger


class TestConversationLogger:
    def test_log_creates_daily_file(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "안녕")
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1

    def test_log_appends_to_existing(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "안녕")
        logger.log("assistant", "야 뭐해")
        entries = logger.read_today()
        assert len(entries) == 2

    def test_log_includes_timestamp(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "테스트")
        entries = logger.read_today()
        assert "timestamp" in entries[0]

    def test_log_stores_role_and_content(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "질문입니다")
        entries = logger.read_today()
        assert entries[0]["role"] == "user"
        assert entries[0]["content"] == "질문입니다"

    def test_log_dir_created_if_not_exists(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        logger = ConversationLogger(log_dir=nested)
        logger.log("user", "test")
        assert nested.exists()
        files = list(nested.glob("*.jsonl"))
        assert len(files) == 1

    def test_daily_file_name_format(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "hello")
        files = list(tmp_path.glob("*.jsonl"))
        # File name should match YYYY-MM-DD.jsonl pattern
        name = files[0].stem
        parts = name.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year
        assert len(parts[1]) == 2  # month
        assert len(parts[2]) == 2  # day

    def test_log_preserves_unicode(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "한국어 테스트 메시지")
        entries = logger.read_today()
        assert entries[0]["content"] == "한국어 테스트 메시지"

    def test_each_line_is_valid_json(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "one")
        logger.log("assistant", "two")
        logger.log("user", "three")
        files = list(tmp_path.glob("*.jsonl"))
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            entry = json.loads(line)
            assert "timestamp" in entry
            assert "role" in entry
            assert "content" in entry

    def test_read_today_empty_if_no_file(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        assert logger.read_today() == []
