import asyncio

_VALID_FORMATS = {"s16le", "s16be", "u8", "float32le", "float32be"}
_VALID_CHANNELS = {1, 2}


class AudioPlayer:
    """Async wrapper around paplay for PCM audio output."""

    def __init__(
        self, rate: int = 16000, channels: int = 1, format: str = "s16le"
    ) -> None:
        if format not in _VALID_FORMATS:
            raise ValueError(f"Invalid format: {format}")
        if channels not in _VALID_CHANNELS:
            raise ValueError(f"Invalid channels: {channels}")
        if not (8000 <= rate <= 192000):
            raise ValueError(f"Invalid rate: {rate}")
        self._rate = rate
        self._channels = channels
        self._format = format
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start paplay subprocess."""
        self._process = await asyncio.create_subprocess_exec(
            "paplay",
            f"--format={self._format}",
            f"--rate={self._rate}",
            f"--channels={self._channels}",
            "--raw",
            stdin=asyncio.subprocess.PIPE,
        )

    async def play_chunk(self, data: bytes) -> None:
        """Write audio chunk to paplay stdin."""
        if self._process and self._process.stdin:
            self._process.stdin.write(data)
            await self._process.stdin.drain()

    async def finish(self) -> None:
        """Close stdin and wait for paplay to finish."""
        if self._process and self._process.stdin:
            self._process.stdin.close()
            await self._process.wait()
        self._process = None

    async def kill(self) -> None:
        """Immediately kill paplay (for barge-in)."""
        try:
            if self._process and self._process.returncode is None:
                self._process.kill()
                await self._process.wait()
        finally:
            self._process = None
