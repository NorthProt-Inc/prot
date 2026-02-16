import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
class TestApp:
    async def test_health_endpoint(self):
        mock_pipeline = MagicMock()
        mock_pipeline.startup = AsyncMock()
        mock_pipeline.shutdown = AsyncMock()
        mock_pipeline.state.state.value = "idle"
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
        mock_pipeline.state.state.value = "listening"
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
