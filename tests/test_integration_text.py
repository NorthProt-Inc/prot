import pytest
from prot.llm import LLMClient
from prot.context import ContextManager
from prot.processing import chunk_sentences
from prot.persona import load_persona

@pytest.mark.integration
@pytest.mark.asyncio
async def test_text_to_text_pipeline():
    """Full text pipeline: user input -> Claude -> sanitized output."""
    persona = load_persona()
    cm = ContextManager(persona_text=persona)
    cm.add_message("user", "야 오늘 뭐하냐")

    client = LLMClient()
    raw = ""
    async for chunk in client.stream_response(
        system_blocks=cm.build_system_blocks(),
        tools=cm.build_tools(),
        messages=cm.get_messages(),
    ):
        raw += chunk

    assert len(raw) > 0
    sentences, remainder = chunk_sentences(raw)
    assert len(sentences) + (1 if remainder else 0) >= 1
    print(f"\nAxel: {raw}")
