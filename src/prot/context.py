from datetime import datetime
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Vancouver")


class ContextManager:
    """Build and manage the multi-block system prompt for Claude API calls.

    The 3-block layout is designed for optimal prompt caching:
      Block 1: Persona + Rules      (STATIC, cached)
      Block 2: GraphRAG Context      (TOPIC-DEPENDENT, cached)
      Block 3: Dynamic Context       (PER-REQUEST, NOT cached, MUST be last)

    Placing dynamic content last preserves the cached prefix so that
    blocks 1 and 2 can hit the prompt cache on subsequent requests.
    """

    def __init__(self, persona_text: str, rag_context: str = "") -> None:
        self._persona = persona_text
        self._rag_context = rag_context
        self._messages: list[dict] = []

    def build_system_blocks(self) -> list[dict]:
        """Build 3-block system prompt with cache control.

        Order is CRITICAL for prompt caching:
          Block 1: Persona + Rules (STATIC, cached)
          Block 2: GraphRAG Context (TOPIC-DEPENDENT, cached)
          Block 3: Dynamic Context (PER-REQUEST, NOT cached, MUST be last)

        If dynamic content (datetime) sits between cached blocks, it breaks the
        cache prefix -- downstream blocks would NEVER hit cache.
        """
        block1_persona: dict = {
            "type": "text",
            "text": self._persona,
            "cache_control": {"type": "ephemeral"},
        }
        block2_rag: dict = {
            "type": "text",
            "text": self._rag_context or "(no additional context)",
            "cache_control": {"type": "ephemeral"},
        }
        block3_dynamic: dict = {
            "type": "text",
            "text": (
                f"datetime: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"timezone: America/Vancouver"
            ),
        }
        return [block1_persona, block2_rag, block3_dynamic]

    def build_tools(self, hass_registry=None) -> list[dict]:
        """Build tool definitions with cache on last tool."""
        web_search: dict = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 1,
            "user_location": {
                "type": "approximate",
                "city": "Vancouver",
                "country": "CA",
                "timezone": "America/Vancouver",
            },
        }
        tools = [web_search]

        if hass_registry is not None:
            tools.extend(hass_registry.build_tool_schemas())

        # Ensure last tool has cache_control
        if "cache_control" not in tools[-1]:
            tools[-1]["cache_control"] = {"type": "ephemeral"}

        return tools

    def add_message(self, role: str, content: str | list) -> None:
        """Append a message. Content can be str or list of content blocks."""
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return a copy of the conversation history."""
        return list(self._messages)

    def get_recent_messages(self, max_turns: int = 10) -> list[dict]:
        """Return recent conversation history (sliding window).

        Keeps the last max_turns * 2 messages, ensuring the window
        starts at a real user message (not an orphaned tool_result).
        """
        max_messages = max_turns * 2
        messages = self._messages[-max_messages:]
        while messages and (
            messages[0]["role"] != "user"
            or (
                isinstance(messages[0]["content"], list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in messages[0]["content"]
                )
            )
        ):
            messages = messages[1:]
        return messages

    def update_rag_context(self, context: str) -> None:
        """Replace the RAG context for the next system prompt build."""
        self._rag_context = context
