import pytest
from starlette.testclient import TestClient


@pytest.mark.integration
def test_chat_websocket_roundtrip():
    """Full WebSocket chat: send message, receive streaming response."""
    from prot.app import app

    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"type": "message", "content": "안녕 악셀"})

        chunks = []
        full_text = None
        for _ in range(100):
            msg = ws.receive_json()
            if msg["type"] == "chunk":
                chunks.append(msg["content"])
            elif msg["type"] == "done":
                full_text = msg["full_text"]
                break
            elif msg["type"] == "error":
                pytest.fail(f"Error: {msg['message']}")

        assert len(chunks) > 0
        assert full_text is not None
        assert len(full_text) > 0
        print(f"\nAxel: {full_text}")
