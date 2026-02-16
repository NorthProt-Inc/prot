import torch
import numpy as np


class VADProcessor:
    """Silero VAD wrapper for speech detection."""

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        speech_count_threshold: int = 3,
    ):
        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._model.eval()
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._speech_count_threshold = speech_count_threshold
        self._speech_count = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """Check if PCM audio chunk contains speech."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)
        prob = self._model(tensor, self._sample_rate).item()

        if prob >= self._threshold:
            self._speech_count += 1
        else:
            self._speech_count = 0

        return self._speech_count >= self._speech_count_threshold

    def reset(self) -> None:
        """Reset internal state."""
        self._speech_count = 0
        self._model.reset_states()
