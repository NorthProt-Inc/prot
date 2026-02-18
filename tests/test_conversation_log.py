"""Tests for ConversationLogger — daily JSONL conversation archival."""

import json
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from prot.conversation_log import ConversationLogger


class TestConversationLogger:
    def test_save_session_creates_jsonl_file(self, tmp_path):
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        path = logger.save_session(session_id, messages)

        assert path is not None
        assert path.exists()
        assert path.suffix == ".jsonl"
        line = path.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["session_id"] == str(session_id)
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "hello"

    def test_save_session_empty_messages_returns_none(self, tmp_path):
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        result = logger.save_session(session_id, [])
        assert result is None

    def test_save_session_daily_file_naming(self, tmp_path):
        """File name is {YYYY-MM-DD}.jsonl with no session_id."""
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("abcdef01-2345-6789-abcd-ef0123456789")
        messages = [{"role": "user", "content": "test"}]
        path = logger.save_session(session_id, messages)

        assert path is not None
        assert path.suffix == ".jsonl"
        # Stem is just the date: YYYY-MM-DD
        parts = path.stem.split("-")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_save_session_appends_to_existing_file(self, tmp_path):
        """Multiple sessions on the same day append to the same file."""
        logger = ConversationLogger(log_dir=str(tmp_path))
        sid1 = UUID("11111111-1111-1111-1111-111111111111")
        sid2 = UUID("22222222-2222-2222-2222-222222222222")

        logger.save_session(sid1, [{"role": "user", "content": "first"}])
        path = logger.save_session(sid2, [{"role": "user", "content": "second"}])

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["session_id"] == str(sid1)
        assert json.loads(lines[1])["session_id"] == str(sid2)

    def test_save_session_list_content_serialized_to_text(self, tmp_path):
        """List content blocks should be serialized as plain text."""
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        messages = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]},
        ]
        path = logger.save_session(session_id, messages)

        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert isinstance(data["messages"][0]["content"], str)
        assert "hello" in data["messages"][0]["content"]

    def test_save_session_object_content_blocks(self, tmp_path):
        """SDK content block objects with .text attribute should serialize."""
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        block = MagicMock()
        block.text = "compacted text"
        messages = [{"role": "assistant", "content": [block]}]
        path = logger.save_session(session_id, messages)

        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert "compacted text" in data["messages"][0]["content"]

    def test_save_session_korean_text(self, tmp_path):
        """Korean text should be preserved (ensure_ascii=False)."""
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        messages = [{"role": "user", "content": "안녕하세요 반갑습니다"}]
        path = logger.save_session(session_id, messages)

        raw = path.read_text(encoding="utf-8")
        assert "안녕하세요" in raw
        data = json.loads(raw.strip())
        assert data["messages"][0]["content"] == "안녕하세요 반갑습니다"

    def test_save_session_creates_parent_dirs(self, tmp_path):
        """Parent directories should be created if missing."""
        nested = tmp_path / "a" / "b" / "c"
        logger = ConversationLogger(log_dir=str(nested))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        messages = [{"role": "user", "content": "test"}]
        path = logger.save_session(session_id, messages)

        assert path is not None
        assert path.exists()
        assert nested.exists()

    def test_each_line_is_valid_json(self, tmp_path):
        """Each line in JSONL must be independently parseable."""
        logger = ConversationLogger(log_dir=str(tmp_path))
        for i in range(5):
            sid = UUID(f"{'0' * 7}{i}-0000-0000-0000-000000000000")
            logger.save_session(sid, [{"role": "user", "content": f"msg {i}"}])

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            data = json.loads(line)
            assert "session_id" in data
            assert "messages" in data
