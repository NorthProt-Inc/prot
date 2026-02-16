import asyncio


class AudioPlayer:
    """Async wrapper around paplay for PCM audio output."""

    def __init__(
        self, rate: int = 16000, channels: int = 1, format: str = "s16le"
    ) -> None:
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
        if self._process and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        self._process = None
