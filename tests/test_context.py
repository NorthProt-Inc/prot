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
