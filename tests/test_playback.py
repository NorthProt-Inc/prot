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

    async def test_finish_closes_stdin_and_waits(self):
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stdin.close = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            player = AudioPlayer()
            await player.start()
            await player.finish()

            mock_proc.stdin.close.assert_called_once()
            mock_proc.wait.assert_called_once()
            assert player._process is None

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid format"):
            AudioPlayer(format="invalid")

    def test_invalid_channels_raises(self):
        with pytest.raises(ValueError, match="Invalid channels"):
            AudioPlayer(channels=99)

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="Invalid rate"):
            AudioPlayer(rate=0)
