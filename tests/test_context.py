import pytest
from unittest.mock import MagicMock
from prot.context import ContextManager


class TestContextManager:
    def test_build_system_blocks_returns_three_blocks(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert len(blocks) == 3

    def test_block1_persona_has_cache_control(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "test persona" in blocks[0]["text"]

    def test_block2_rag_has_cache_control(self):
        cm = ContextManager(persona_text="test persona", rag_context="some context")
        blocks = cm.build_system_blocks()
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}
        assert "some context" in blocks[1]["text"]

    def test_block3_dynamic_has_no_cache_control(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert "cache_control" not in blocks[2]

    def test_block3_contains_datetime(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert "datetime:" in blocks[2]["text"]

    def test_block_order_is_persona_rag_dynamic(self):
        """Verify ordering: Block 1=persona (cached), Block 2=RAG (cached), Block 3=dynamic (NOT cached).
        This order is CRITICAL for prompt caching: dynamic content must be LAST to preserve cached prefix."""
        cm = ContextManager(persona_text="persona text", rag_context="rag text")
        blocks = cm.build_system_blocks()
        assert "persona text" in blocks[0]["text"]
        assert "rag text" in blocks[1]["text"]
        assert "datetime:" in blocks[2]["text"]
        assert "cache_control" in blocks[0]
        assert "cache_control" in blocks[1]
        assert "cache_control" not in blocks[2]

    def test_build_system_blocks_default_channel_is_voice(self):
        cm = ContextManager(persona_text="test persona", rag_context="")
        blocks = cm.build_system_blocks()
        assert "channel: voice" in blocks[2]["text"]

    def test_build_system_blocks_chat_channel(self):
        cm = ContextManager(persona_text="test persona", rag_context="", channel="chat")
        blocks = cm.build_system_blocks()
        assert "channel: chat" in blocks[2]["text"]

    def test_build_system_blocks_voice_channel_explicit(self):
        cm = ContextManager(persona_text="test persona", rag_context="", channel="voice")
        blocks = cm.build_system_blocks()
        assert "channel: voice" in blocks[2]["text"]

    def test_build_tools_without_hass_returns_web_search_only(self):
        cm = ContextManager(persona_text="test", rag_context="")
        tools = cm.build_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0]["name"] == "web_search"

    def test_build_tools_with_hass_agent(self):
        cm = ContextManager(persona_text="test", rag_context="")
        mock_agent = MagicMock()
        mock_agent.build_tool.return_value = {
            "name": "hass_request",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        }
        tools = cm.build_tools(hass_agent=mock_agent)
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        assert names == ["web_search", "hass_request"]

    def test_build_tools_last_tool_has_cache_control(self):
        cm = ContextManager(persona_text="test", rag_context="")
        tools = cm.build_tools()
        assert "cache_control" in tools[-1]

    def test_build_tools_with_hass_last_tool_has_cache_control(self):
        cm = ContextManager(persona_text="test", rag_context="")
        mock_agent = MagicMock()
        mock_agent.build_tool.return_value = {"name": "hass_request", "input_schema": {}}
        tools = cm.build_tools(hass_agent=mock_agent)
        assert "cache_control" in tools[-1]
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}

    def test_build_tools_adds_cache_control_to_hass_tool(self):
        cm = ContextManager(persona_text="test", rag_context="")
        mock_agent = MagicMock()
        mock_agent.build_tool.return_value = {"name": "hass_request", "input_schema": {}}
        tools = cm.build_tools(hass_agent=mock_agent)
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}

    def test_add_message_and_get_history(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "안녕")
        cm.add_message("assistant", "야 뭐해")
        history = cm.get_messages()
        assert len(history) == 2
        assert history[0]["role"] == "user"

    def test_no_sliding_window_all_messages_retained(self):
        """Compact API manages context — no local message trimming."""
        cm = ContextManager(persona_text="test", rag_context="")
        for i in range(100):
            cm.add_message("user", f"msg-{i}")
        assert len(cm.get_messages()) == 100

    def test_add_message_with_list_content(self):
        """Content can be a list of content blocks (compact API compaction)."""
        cm = ContextManager(persona_text="test", rag_context="")
        blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        cm.add_message("assistant", blocks)
        history = cm.get_messages()
        assert len(history) == 1
        assert history[0]["content"] is blocks

    def test_get_messages_mixed_content_types(self):
        """Messages can have str or list content."""
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "hello")
        cm.add_message("assistant", [{"type": "text", "text": "hi"}])
        history = cm.get_messages()
        assert isinstance(history[0]["content"], str)
        assert isinstance(history[1]["content"], list)


class TestBlock3Timeline:
    def test_block3_contains_session_start_after_messages(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "hello")
        blocks = cm.build_system_blocks()
        assert "session_start:" in blocks[2]["text"]

    def test_block3_contains_recent_turns(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "hello")
        cm.add_message("assistant", "hi")
        blocks = cm.build_system_blocks()
        text = blocks[2]["text"]
        assert "recent_turns:" in text
        assert "user" in text
        assert "assistant" in text

    def test_block3_no_timeline_without_messages(self):
        cm = ContextManager(persona_text="test", rag_context="")
        blocks = cm.build_system_blocks()
        assert "session_start:" not in blocks[2]["text"]
        assert "recent_turns:" not in blocks[2]["text"]

    def test_message_times_excludes_tool_results(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "hello")
        cm.add_message("assistant", "thinking...")
        cm.add_message("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}])
        cm.add_message("assistant", "done")
        assert len(cm._message_times) == 3  # user, assistant, assistant (not tool_result)

    def test_recent_turns_limited_to_last_10(self):
        cm = ContextManager(persona_text="test", rag_context="")
        for i in range(20):
            cm.add_message("user", f"msg-{i}")
            cm.add_message("assistant", f"resp-{i}")
        blocks = cm.build_system_blocks()
        lines = [l for l in blocks[2]["text"].split("\n") if l.strip().startswith("- ")]
        assert len(lines) == 10


class TestGetRecentMessages:
    """get_recent_messages() — return all messages with orphan boundary correction."""

    def test_returns_all_messages(self):
        cm = ContextManager(persona_text="test", rag_context="")
        for i in range(50):
            cm.add_message("user", f"msg-{i}")
            cm.add_message("assistant", f"resp-{i}")
        result = cm.get_recent_messages()
        assert len(result) == 100

    def test_starts_with_user_role(self):
        cm = ContextManager(persona_text="test", rag_context="")
        # Simulate orphaned assistant at the start (shouldn't happen but defensive)
        cm._messages = [
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "real"},
            {"role": "assistant", "content": "resp"},
        ]
        result = cm.get_recent_messages()
        assert result[0]["role"] == "user"

    def test_skips_orphan_tool_result_at_start(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "sunny"}])
        cm.add_message("assistant", "It's sunny!")
        cm.add_message("user", "thanks")
        cm.add_message("assistant", "welcome")
        result = cm.get_recent_messages()
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], str)

    def test_preserves_user_message_with_non_tool_result_list(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", [{"type": "text", "text": "hello with blocks"}])
        cm.add_message("assistant", "response")
        result = cm.get_recent_messages()
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_empty_history(self):
        cm = ContextManager(persona_text="test", rag_context="")
        assert cm.get_recent_messages() == []
