import pyaudio
from collections.abc import Callable


class AudioManager:
    """PyAudio mic input with non-blocking callback."""

    def __init__(
        self,
        device_index: int = 11,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        on_audio: Callable[[bytes], None] | None = None,
    ) -> None:
        self._pa = pyaudio.PyAudio()
        self._device_index = device_index
        self._sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._on_audio = on_audio
        self._stream: pyaudio.Stream | None = None

    def start(self) -> None:
        """Open mic stream with callback."""
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        """Stop and close mic stream and terminate PyAudio."""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        self._pa.terminate()

    def _audio_callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict,
        status: int,
    ) -> tuple[None, int]:
        """PyAudio stream callback — forwards data to on_audio handler."""
        if self._on_audio and in_data:
            self._on_audio(in_data)
        return (None, pyaudio.paContinue)

