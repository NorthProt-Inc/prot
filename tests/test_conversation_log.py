"""Tests for ConversationLogger — daily JSON conversation archival."""

import json
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from prot.conversation_log import ConversationLogger


class TestConversationLogger:
    def test_save_session_creates_json_file(self, tmp_path):
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        path = logger.save_session(session_id, messages)

        assert path is not None
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["session_id"] == str(session_id)
        assert data["version"] == "1.0"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "hello"

    def test_save_session_empty_messages_returns_none(self, tmp_path):
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        result = logger.save_session(session_id, [])
        assert result is None

    def test_save_session_file_naming_pattern(self, tmp_path):
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("abcdef01-2345-6789-abcd-ef0123456789")
        messages = [{"role": "user", "content": "test"}]
        path = logger.save_session(session_id, messages)

        assert path is not None
        # File name: {date}-{session_id[:8]}.json
        assert path.name.endswith("-abcdef01.json")
        assert path.suffix == ".json"

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

        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data["messages"][0]["content"], str)
        assert "hello" in data["messages"][0]["content"]
        assert "world" in data["messages"][0]["content"]

    def test_save_session_object_content_blocks(self, tmp_path):
        """SDK content block objects with .text attribute should serialize."""
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        block = MagicMock()
        block.text = "compacted text"
        messages = [{"role": "assistant", "content": [block]}]
        path = logger.save_session(session_id, messages)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "compacted text" in data["messages"][0]["content"]

    def test_save_session_korean_text(self, tmp_path):
        """Korean text should be preserved (ensure_ascii=False)."""
        logger = ConversationLogger(log_dir=str(tmp_path))
        session_id = UUID("12345678-1234-1234-1234-123456789abc")
        messages = [{"role": "user", "content": "안녕하세요 반갑습니다"}]
        path = logger.save_session(session_id, messages)

        raw = path.read_text(encoding="utf-8")
        assert "안녕하세요" in raw
        data = json.loads(raw)
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
