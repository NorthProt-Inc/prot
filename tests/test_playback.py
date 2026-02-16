import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prot.playback import AudioPlayer


@pytest.mark.asyncio
class TestAudioPlayer:
    async def test_play_chunk_writes_to_stdin(self):
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            player = AudioPlayer()
            await player.start()
            await player.play_chunk(b"\x00" * 1024)
            mock_proc.stdin.write.assert_called_once_with(b"\x00" * 1024)

    async def test_kill_terminates_process(self):
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            player = AudioPlayer()
            await player.start()
            await player.kill()
            mock_proc.kill.assert_called_once()
