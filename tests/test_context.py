import pytest
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

    def test_build_tools_returns_list(self):
        cm = ContextManager(persona_text="test", rag_context="")
        tools = cm.build_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1  # at least web_search

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


class TestGetRecentMessages:
    """get_recent_messages() — sliding window over conversation history."""

    def test_returns_all_when_under_limit(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "msg1")
        cm.add_message("assistant", "resp1")
        result = cm.get_recent_messages(max_turns=10)
        assert len(result) == 2

    def test_trims_to_max_turns(self):
        cm = ContextManager(persona_text="test", rag_context="")
        for i in range(20):
            cm.add_message("user", f"msg-{i}")
            cm.add_message("assistant", f"resp-{i}")
        result = cm.get_recent_messages(max_turns=5)
        assert len(result) == 10
        assert result[0]["content"] == "msg-15"
        assert result[-1]["content"] == "resp-19"

    def test_starts_with_user_role(self):
        cm = ContextManager(persona_text="test", rag_context="")
        for i in range(10):
            cm.add_message("user", f"msg-{i}")
            cm.add_message("assistant", f"resp-{i}")
        result = cm.get_recent_messages(max_turns=3)
        assert result[0]["role"] == "user"

    def test_skips_orphan_tool_result_at_boundary(self):
        cm = ContextManager(persona_text="test", rag_context="")
        cm.add_message("user", "hello")
        cm.add_message("assistant", "hi")
        cm.add_message("user", "weather?")
        cm.add_message("assistant", "checking...")
        cm.add_message("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "sunny"}])
        cm.add_message("assistant", "It's sunny!")
        cm.add_message("user", "thanks")
        cm.add_message("assistant", "welcome")
        result = cm.get_recent_messages(max_turns=2)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], str)

    def test_get_messages_still_returns_all(self):
        cm = ContextManager(persona_text="test", rag_context="")
        for i in range(50):
            cm.add_message("user", f"msg-{i}")
        assert len(cm.get_messages()) == 50

    def test_empty_history(self):
        cm = ContextManager(persona_text="test", rag_context="")
        assert cm.get_recent_messages(max_turns=10) == []
