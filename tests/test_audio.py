from unittest.mock import MagicMock, patch

from prot.audio import AudioManager


class TestAudioManager:
    def test_chunk_size_config(self):
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11, chunk_size=512)
            assert mgr.chunk_size == 512

    def test_device_index_config(self):
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=7)
            assert mgr._device_index == 7

    def test_sample_rate_default(self):
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11)
            assert mgr._sample_rate == 16000

    def test_callback_receives_data(self):
        received = []
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11, on_audio=lambda d: received.append(d))
            mgr._audio_callback(b"\x00" * 1024, 512, {}, 0)
            assert len(received) == 1
            assert received[0] == b"\x00" * 1024

    def test_callback_ignores_none_data(self):
        received = []
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11, on_audio=lambda d: received.append(d))
            result = mgr._audio_callback(None, 512, {}, 0)
            assert len(received) == 0
            assert result == (None, 0)  # pyaudio.paContinue == 0

    def test_audio_callback_returns_protocol_tuple(self):
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11)
            result = mgr._audio_callback(b"\x00" * 512, 256, {}, 0)
            assert result == (None, 0)  # pyaudio.paContinue == 0

    def test_stop_cleans_up_stream(self):
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11)
            mock_stream = MagicMock()
            mgr._stream = mock_stream
            mgr.stop()
            mock_stream.stop_stream.assert_called_once()
            mock_stream.close.assert_called_once()
            assert mgr._stream is None

    def test_stop_noop_when_no_stream(self):
        with patch("prot.audio.pyaudio.PyAudio"):
            mgr = AudioManager(device_index=11)
            mgr.stop()  # should not raise
            assert mgr._stream is None

    def test_start_opens_stream(self):
        with patch("prot.audio.pyaudio.PyAudio") as mock_pa_cls:
            mock_pa = mock_pa_cls.return_value
            mock_stream = MagicMock()
            mock_pa.open.return_value = mock_stream

            mgr = AudioManager(device_index=11, chunk_size=512)
            mgr.start()

            mock_pa.open.assert_called_once()
            kwargs = mock_pa.open.call_args
            assert kwargs.kwargs["input"] is True
            assert kwargs.kwargs["input_device_index"] == 11
            assert kwargs.kwargs["frames_per_buffer"] == 512
            assert kwargs.kwargs["channels"] == 1
            mock_stream.start_stream.assert_called_once()
            assert mgr._stream is mock_stream

    def test_stop_terminates_pyaudio(self):
        with patch("prot.audio.pyaudio.PyAudio") as mock_pa_cls:
            mock_pa = mock_pa_cls.return_value
            mgr = AudioManager(device_index=11)
            mgr.stop()
            mock_pa.terminate.assert_called_once()
