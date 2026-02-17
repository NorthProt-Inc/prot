import torch


class VADProcessor:
    """Silero VAD wrapper for speech detection."""

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        speech_count_threshold: int = 3,
        chunk_bytes: int = 512,
    ):
        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._model.eval()
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._speech_count_threshold = speech_count_threshold
        self._speech_count = 0
        # Pre-allocate float32 buffer (int16 = 2 bytes/sample)
        self._float_buf = torch.empty(chunk_bytes // 2, dtype=torch.float32)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """Check if PCM audio chunk contains speech."""
        raw = torch.frombuffer(pcm_bytes, dtype=torch.int16)
        n = raw.numel()
        if n > self._float_buf.numel():
            self._float_buf = torch.empty(n, dtype=torch.float32)
        buf = self._float_buf[:n]
        buf.copy_(raw)
        buf.div_(32768.0)
        prob = self._model(buf, self._sample_rate).item()

        if prob >= self._threshold:
            self._speech_count += 1
        else:
            self._speech_count = 0

        return self._speech_count >= self._speech_count_threshold

    def reset(self) -> None:
        """Reset internal state."""
        self._speech_count = 0
        self._model.reset_states()
