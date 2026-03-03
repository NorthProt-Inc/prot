import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient

from prot.engine import ConversationEngine, ResponseResult, ToolIterationMarker


@pytest.mark.asyncio
class TestApp:
    async def test_health_endpoint(self):
        mock_pipeline = MagicMock()
        mock_pipeline.startup = AsyncMock()
        mock_pipeline.shutdown = AsyncMock()
        mock_pipeline.current_state = "idle"
        mock_pipeline.on_audio_chunk = MagicMock()

        mock_audio = MagicMock()

        with patch("prot.app.Pipeline", return_value=mock_pipeline), \
             patch("prot.app.AudioManager", return_value=mock_audio):
            import prot.app as app_module
            app_module.pipeline = mock_pipeline
            app_module.audio = mock_audio
            transport = ASGITransport(app=app_module.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["state"] == "idle"
            # Cleanup
            app_module.pipeline = None
            app_module.audio = None

    async def test_state_endpoint(self):
        mock_pipeline = MagicMock()
        mock_pipeline.startup = AsyncMock()
        mock_pipeline.shutdown = AsyncMock()
        mock_pipeline.current_state = "listening"
        mock_pipeline.on_audio_chunk = MagicMock()

        mock_audio = MagicMock()

        with patch("prot.app.Pipeline", return_value=mock_pipeline), \
             patch("prot.app.AudioManager", return_value=mock_audio):
            import prot.app as app_module
            app_module.pipeline = mock_pipeline
            app_module.audio = mock_audio
            transport = ASGITransport(app=app_module.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/state")
            assert response.status_code == 200
            assert response.json()["state"] == "listening"
            # Cleanup
            app_module.pipeline = None
            app_module.audio = None


class TestChatWebSocket:
    def test_chat_sends_and_receives(self):
        """Sync test using Starlette TestClient for WebSocket."""
        mock_engine = MagicMock(spec=ConversationEngine)
        mock_engine.busy = False
        mock_engine.add_user_message = MagicMock()
        mock_engine.shutdown_summarize = AsyncMock()
        mock_engine.last_result = ResponseResult(
            full_text="Hello world", interrupted=False
        )

        async def fake_respond():
            yield "Hello "
            yield "world"

        mock_engine.respond = fake_respond

        mock_pipeline = MagicMock()
        mock_pipeline.startup = AsyncMock()
        mock_pipeline.shutdown = AsyncMock()
        mock_pipeline._llm = MagicMock()
        mock_pipeline._hass_agent = None
        mock_pipeline._memory = None
        mock_pipeline._graphrag = None
        mock_pipeline.current_state = "idle"

        mock_audio = MagicMock()
        mock_audio.start = MagicMock()
        mock_audio.stop = MagicMock()

        with patch("prot.app.Pipeline", return_value=mock_pipeline), \
             patch("prot.app.AudioManager", return_value=mock_audio), \
             patch("prot.app.ConversationEngine", return_value=mock_engine) as mock_engine_cls, \
             patch("prot.app.load_persona", return_value="test persona"):
            import prot.app as app_module
            app_module.pipeline = mock_pipeline
            app_module.audio = mock_audio

            client = TestClient(app_module.app)
            with client.websocket_connect("/chat") as ws:
                ws.send_json({"type": "message", "content": "hello"})
                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] in ("done", "error"):
                        break

            chunk_msgs = [m for m in messages if m["type"] == "chunk"]
            done_msg = [m for m in messages if m["type"] == "done"][0]

            assert len(chunk_msgs) == 2
            assert chunk_msgs[0]["content"] == "Hello "
            assert chunk_msgs[1]["content"] == "world"
            assert done_msg["full_text"] == "Hello world"

            mock_engine.add_user_message.assert_called_once_with("hello")

            # Verify channel="chat" was passed
            mock_engine_cls.assert_called_once()
            call_kwargs = mock_engine_cls.call_args
            assert call_kwargs.kwargs.get("ctx") is not None

            # Cleanup
            app_module.pipeline = None
            app_module.audio = None
