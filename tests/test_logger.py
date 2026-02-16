import json
from pathlib import Path

import pytest

from prot.logger import ConversationLogger


class TestConversationLogger:
    def test_log_creates_daily_file(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "안녕")
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_log_appends_to_existing(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "안녕")
        logger.log("assistant", "야 뭐해")
        files = list(tmp_path.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert len(data) == 2

    def test_log_includes_timestamp(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "테스트")
        files = list(tmp_path.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert "timestamp" in data[0]

    def test_log_stores_role_and_content(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "질문입니다")
        files = list(tmp_path.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert data[0]["role"] == "user"
        assert data[0]["content"] == "질문입니다"

    def test_log_dir_created_if_not_exists(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        logger = ConversationLogger(log_dir=nested)
        logger.log("user", "test")
        assert nested.exists()
        files = list(nested.glob("*.json"))
        assert len(files) == 1

    def test_daily_file_name_format(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "hello")
        files = list(tmp_path.glob("*.json"))
        # File name should match YYYY-MM-DD.json pattern
        name = files[0].stem
        parts = name.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year
        assert len(parts[1]) == 2  # month
        assert len(parts[2]) == 2  # day

    def test_log_preserves_unicode(self, tmp_path):
        logger = ConversationLogger(log_dir=tmp_path)
        logger.log("user", "한국어 테스트 메시지")
        files = list(tmp_path.glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data[0]["content"] == "한국어 테스트 메시지"
