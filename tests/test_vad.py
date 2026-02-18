import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call


class TestVADProcessor:
    @pytest.fixture(autouse=True)
    def _mock_torch_hub(self):
        """Mock torch.hub.load to avoid downloading the actual model."""
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(
            item=MagicMock(return_value=0.1)
        )  # low probability = no speech
        with patch("prot.vad.torch.hub.load", return_value=(mock_model, None)):
            self._mock_model = mock_model
            yield

    def test_silence_returns_false(self):
        from prot.vad import VADProcessor

        vad = VADProcessor()
        silence = np.zeros(512, dtype=np.int16)
        assert vad.is_speech(silence.tobytes()) is False

    def test_threshold_property(self):
        from prot.vad import VADProcessor

        vad = VADProcessor(threshold=0.5)
        assert vad.threshold == 0.5
        vad.threshold = 0.8
        assert vad.threshold == 0.8

    def test_reset_clears_state(self):
        from prot.vad import VADProcessor

        vad = VADProcessor()
        vad._speech_count = 5
        vad.reset()
        assert vad._speech_count == 0

    def test_reset_calls_model_reset_states(self):
        from prot.vad import VADProcessor

        vad = VADProcessor()
        vad.reset()
        self._mock_model.reset_states.assert_called_once()

    def test_speech_detected_after_threshold_count(self):
        """Test that speech is detected after speech_count_threshold consecutive detections."""
        from prot.vad import VADProcessor

        # Make the mock return high probability (speech detected)
        self._mock_model.return_value = MagicMock(
            item=MagicMock(return_value=0.9)
        )
        vad = VADProcessor(threshold=0.5, speech_count_threshold=3)
        audio = np.ones(512, dtype=np.int16)
        pcm = audio.tobytes()
        # First two calls: below threshold count
        assert vad.is_speech(pcm) is False
        assert vad.is_speech(pcm) is False
        # Third call: reaches threshold
        assert vad.is_speech(pcm) is True

    def test_speech_count_resets_on_silence(self):
        """Test that speech_count resets when a non-speech frame arrives."""
        from prot.vad import VADProcessor

        self._mock_model.return_value = MagicMock(
            item=MagicMock(return_value=0.9)
        )
        vad = VADProcessor(threshold=0.5, speech_count_threshold=3)
        audio = np.ones(512, dtype=np.int16)
        pcm = audio.tobytes()
        vad.is_speech(pcm)  # count=1
        vad.is_speech(pcm)  # count=2
        # Now simulate silence
        self._mock_model.return_value = MagicMock(
            item=MagicMock(return_value=0.1)
        )
        vad.is_speech(pcm)  # should reset count
        assert vad._speech_count == 0

    def test_model_eval_called_on_init(self):
        """Test that model.eval() is called during initialization."""
        from prot.vad import VADProcessor

        VADProcessor()
        self._mock_model.eval.assert_called_once()

    def test_default_sample_rate(self):
        from prot.vad import VADProcessor

        vad = VADProcessor()
        assert vad._sample_rate == 16000

    def test_custom_speech_count_threshold(self):
        """Test that a custom speech_count_threshold is respected."""
        from prot.vad import VADProcessor

        self._mock_model.return_value = MagicMock(
            item=MagicMock(return_value=0.9)
        )
        vad = VADProcessor(threshold=0.5, speech_count_threshold=5)
        audio = np.ones(512, dtype=np.int16)
        pcm = audio.tobytes()
        for _ in range(4):
            assert vad.is_speech(pcm) is False
        assert vad.is_speech(pcm) is True

    def test_is_speech_uses_inference_mode(self):
        """is_speech() must run under torch.inference_mode() to prevent autograd graph leaks."""
        from prot.vad import VADProcessor
        import torch

        vad = VADProcessor()
        audio = np.zeros(512, dtype=np.int16)

        with patch("prot.vad.torch.inference_mode") as mock_inf:
            ctx = MagicMock()
            mock_inf.return_value = ctx
            ctx.__enter__ = MagicMock(return_value=None)
            ctx.__exit__ = MagicMock(return_value=False)
            vad.is_speech(audio.tobytes())
            mock_inf.assert_called_once()
            ctx.__enter__.assert_called_once()
            ctx.__exit__.assert_called_once()

    def test_default_chunk_bytes_is_1024(self):
        """Default chunk_bytes should be 1024 (512 samples × 2 bytes)."""
        from prot.vad import VADProcessor

        vad = VADProcessor()
        # float_buf should be 1024 // 2 = 512 samples
        assert vad._float_buf.numel() == 512

    def test_prebuffer_stores_chunks(self):
        from prot.vad import VADProcessor

        vad = VADProcessor(threshold=0.5, prebuffer_chunks=5)
        for i in range(3):
            vad.is_speech(bytes([i]) * 1024)
        assert len(vad.prebuffer) == 3

    def test_prebuffer_rolls_over(self):
        from prot.vad import VADProcessor

        vad = VADProcessor(threshold=0.5, prebuffer_chunks=3)
        for i in range(5):
            vad.is_speech(bytes([i]) * 1024)
        assert len(vad.prebuffer) == 3
        assert vad.prebuffer[0] == bytes([2]) * 1024

    def test_drain_prebuffer_returns_and_clears(self):
        from prot.vad import VADProcessor

        vad = VADProcessor(threshold=0.5, prebuffer_chunks=5)
        for i in range(3):
            vad.is_speech(bytes([i]) * 1024)
        chunks = vad.drain_prebuffer()
        assert len(chunks) == 3
        assert len(vad.prebuffer) == 0

    def test_reset_clears_prebuffer(self):
        from prot.vad import VADProcessor

        vad = VADProcessor(threshold=0.5, prebuffer_chunks=5)
        vad.is_speech(b"\x00" * 1024)
        vad.reset()
        assert len(vad.prebuffer) == 0

    def test_prebuffer_default_size(self):
        from prot.vad import VADProcessor

        vad = VADProcessor(threshold=0.5)
        assert vad.prebuffer.maxlen == 8
