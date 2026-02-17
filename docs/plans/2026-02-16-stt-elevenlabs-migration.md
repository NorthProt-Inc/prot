# STT Migration: Deepgram → ElevenLabs Scribe v2 Realtime

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Deepgram Nova-3 WebSocket STT with ElevenLabs Scribe v2 Realtime WebSocket STT, keeping the same `STTClient` interface so `pipeline.py` needs zero changes.

**Architecture:** Drop-in replacement of `stt.py` internals. The public interface (`connect()`, `send_audio(bytes)`, `disconnect()`, callbacks `on_transcript(text, is_final)` and `on_utterance_end()`) stays identical. ElevenLabs realtime STT uses raw `websockets` (already installed v16.0) with JSON messages over `wss://api.elevenlabs.io/v1/speech-to-text/realtime`. VAD-based commit strategy replaces Deepgram's utterance-end event.

**Tech Stack:** `websockets` 16.0, `elevenlabs` 2.35.0 (API key only — no SDK STT class used), Python 3.12, pytest-asyncio

**Current Deepgram touchpoints (all must be addressed):**
- `src/prot/stt.py` — full rewrite (core change)
- `src/prot/config.py:7,19-23` — `deepgram_api_key`, `deepgram_model`, `deepgram_language`, `deepgram_endpointing`, `deepgram_utterance_end_ms`
- `tests/test_stt.py` — full rewrite (mocks change entirely)
- `tests/conftest.py:8` — `DEEPGRAM_API_KEY` env var
- `pyproject.toml:15` — `deepgram-sdk>=4.0` dependency

---

### Task 1: Rewrite STT client — failing tests first

**Files:**
- Rewrite: `tests/test_stt.py`
- Reference: `src/prot/stt.py` (current interface to preserve)

**Step 1: Write the new test file**

Replace `tests/test_stt.py` entirely. These tests mock `websockets.connect` instead of `AsyncDeepgramClient`. The test design mirrors the current test coverage but targets the new ElevenLabs protocol.

```python
"""Tests for prot.stt — ElevenLabs Scribe v2 Realtime WebSocket STT client."""

import asyncio
import base64
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prot.stt import STTClient


def _make_ws_mock():
    """Create a mock WebSocket that returns session_started then blocks."""
    ws = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=False)
    session_msg = json.dumps({
        "message_type": "session_started",
        "session_id": "test-session",
        "config": {},
    })
    # First recv returns session_started, then block forever
    ws.recv = AsyncMock(side_effect=[session_msg, asyncio.CancelledError()])
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.mark.asyncio
class TestSTTClient:
    async def test_send_audio_encodes_base64_json(self):
        ws = _make_ws_mock()
        with patch("prot.stt.websockets.connect", return_value=ws):
            client = STTClient(api_key="test")
            await client.connect()
            await client.send_audio(b"\x00" * 512)

        call_args = ws.send.call_args[0][0]
        msg = json.loads(call_args)
        assert msg["message_type"] == "input_audio_chunk"
        assert msg["sample_rate"] == 16000
        assert msg["commit"] is False
        raw = base64.b64decode(msg["audio_base_64"])
        assert raw == b"\x00" * 512

    async def test_send_audio_noop_when_no_connection(self):
        client = STTClient(api_key="test")
        await client.send_audio(b"\x00" * 512)  # should not raise

    async def test_partial_transcript_callback(self):
        transcripts = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        ws = _make_ws_mock()
        partial_msg = json.dumps({
            "message_type": "partial_transcript",
            "text": "안녕하세요",
        })
        ws.recv = AsyncMock(side_effect=[
            json.dumps({"message_type": "session_started", "session_id": "s", "config": {}}),
            partial_msg,
            asyncio.CancelledError(),
        ])

        with patch("prot.stt.websockets.connect", return_value=ws):
            client = STTClient(api_key="test", on_transcript=on_transcript)
            await client.connect()
            await asyncio.sleep(0.05)  # let recv_loop process
            await client.disconnect()

        assert ("안녕하세요", False) in transcripts

    async def test_committed_transcript_callback(self):
        transcripts = []
        utt_ends = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        async def on_utt_end():
            utt_ends.append(True)

        ws = _make_ws_mock()
        committed_msg = json.dumps({
            "message_type": "committed_transcript",
            "text": "테스트 문장입니다",
        })
        ws.recv = AsyncMock(side_effect=[
            json.dumps({"message_type": "session_started", "session_id": "s", "config": {}}),
            committed_msg,
            asyncio.CancelledError(),
        ])

        with patch("prot.stt.websockets.connect", return_value=ws):
            client = STTClient(
                api_key="test",
                on_transcript=on_transcript,
                on_utterance_end=on_utt_end,
            )
            await client.connect()
            await asyncio.sleep(0.05)
            await client.disconnect()

        assert ("테스트 문장입니다", True) in transcripts
        assert utt_ends == [True]

    async def test_empty_transcript_skipped(self):
        transcripts = []

        async def on_transcript(text, is_final):
            transcripts.append((text, is_final))

        ws = _make_ws_mock()
        ws.recv = AsyncMock(side_effect=[
            json.dumps({"message_type": "session_started", "session_id": "s", "config": {}}),
            json.dumps({"message_type": "partial_transcript", "text": ""}),
            json.dumps({"message_type": "committed_transcript", "text": ""}),
            asyncio.CancelledError(),
        ])

        with patch("prot.stt.websockets.connect", return_value=ws):
            client = STTClient(api_key="test", on_transcript=on_transcript)
            await client.connect()
            await asyncio.sleep(0.05)
            await client.disconnect()

        assert transcripts == []

    async def test_disconnect_cancels_recv_task(self):
        ws = _make_ws_mock()
        # Block forever on recv after session_started
        ws.recv = AsyncMock(side_effect=[
            json.dumps({"message_type": "session_started", "session_id": "s", "config": {}}),
            asyncio.Future(),  # blocks forever
        ])

        with patch("prot.stt.websockets.connect", return_value=ws):
            client = STTClient(api_key="test")
            await client.connect()
            assert client._recv_task is not None
            await client.disconnect()
            assert client._recv_task is None
            assert client._ws is None

    async def test_disconnect_noop_when_not_connected(self):
        client = STTClient(api_key="test")
        await client.disconnect()  # should not raise

    async def test_no_callbacks_no_error(self):
        ws = _make_ws_mock()
        ws.recv = AsyncMock(side_effect=[
            json.dumps({"message_type": "session_started", "session_id": "s", "config": {}}),
            json.dumps({"message_type": "committed_transcript", "text": "hello"}),
            asyncio.CancelledError(),
        ])

        with patch("prot.stt.websockets.connect", return_value=ws):
            client = STTClient(api_key="test")
            await client.connect()
            await asyncio.sleep(0.05)
            await client.disconnect()

    async def test_send_audio_disconnect_on_failure(self):
        ws = _make_ws_mock()
        ws.send = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("prot.stt.websockets.connect", return_value=ws):
            client = STTClient(api_key="test")
            await client.connect()
            client.disconnect = AsyncMock()
            await client.send_audio(b"\x00" * 512)
            client.disconnect.assert_awaited_once()

    async def test_connect_reconnects_if_already_connected(self):
        ws = _make_ws_mock()
        with patch("prot.stt.websockets.connect", return_value=ws) as mock_connect:
            client = STTClient(api_key="test")
            await client.connect()
            await client.connect()  # should disconnect first, then reconnect
            assert mock_connect.call_count == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_stt.py -v 2>&1 | head -40`
Expected: ImportError or AttributeError — old `stt.py` doesn't have `websockets.connect` or `_ws` attribute.

---

### Task 2: Implement ElevenLabs STT client

**Files:**
- Rewrite: `src/prot/stt.py`

**Step 1: Rewrite `stt.py` with ElevenLabs WebSocket protocol**

```python
"""ElevenLabs Scribe v2 Realtime WebSocket streaming STT client."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from prot.config import settings
from prot.log import get_logger

logger = get_logger(__name__)

_WS_BASE = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


class STTClient:
    """ElevenLabs Scribe v2 Realtime WebSocket streaming client."""

    def __init__(
        self,
        api_key: str | None = None,
        on_transcript: Callable[[str, bool], Awaitable[None]] | None = None,
        on_utterance_end: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._api_key = api_key or settings.elevenlabs_api_key
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Open WebSocket connection to ElevenLabs and start receiving."""
        await self.disconnect()
        try:
            url = (
                f"{_WS_BASE}"
                f"?model_id=scribe_v2_realtime"
                f"&language_code={settings.stt_language}"
                f"&audio_format=pcm_{settings.sample_rate}"
                f"&commit_strategy=vad"
            )
            headers = {"xi-api-key": self._api_key}
            self._ws = await websockets.connect(url, additional_headers=headers)

            # Wait for session_started
            raw = await self._ws.recv()
            session = json.loads(raw)
            if session.get("message_type") != "session_started":
                logger.warning("Unexpected first message", msg=session)

            logger.info("Connected", model="scribe_v2_realtime", lang=settings.stt_language)
            self._recv_task = asyncio.create_task(self._recv_loop())
        except Exception:
            logger.exception("STT connect failed")
            self._ws = None

    async def send_audio(self, data: bytes) -> None:
        """Send PCM audio chunk to ElevenLabs."""
        if self._ws is None:
            return
        try:
            msg = json.dumps({
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(data).decode(),
                "commit": False,
                "sample_rate": settings.sample_rate,
            })
            await self._ws.send(msg)
        except Exception:
            logger.warning("Send failed, disconnecting")
            try:
                await self.disconnect()
            except Exception:
                self._ws = None

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _recv_loop(self) -> None:
        """Receive and dispatch messages from ElevenLabs."""
        ws = self._ws
        while ws:
            try:
                raw = await ws.recv()
            except (ConnectionClosedOK, ConnectionClosedError):
                logger.debug("STT websocket closed")
                break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("STT recv error")
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message from STT")
                continue

            msg_type = msg.get("message_type", "")

            if msg_type == "partial_transcript":
                text = msg.get("text", "")
                if text:
                    await self._fire_transcript(text, is_final=False)

            elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
                text = msg.get("text", "")
                if text:
                    await self._fire_transcript(text, is_final=True)
                    await self._fire_utterance_end()

            elif msg_type in ("error", "auth_error", "input_error"):
                logger.error("STT error", error=msg.get("error", ""))

    async def _fire_transcript(self, text: str, is_final: bool) -> None:
        """Invoke the transcript callback if registered."""
        if self._on_transcript:
            await self._on_transcript(text, is_final)

    async def _fire_utterance_end(self) -> None:
        """Invoke the utterance end callback if registered."""
        if self._on_utterance_end:
            await self._on_utterance_end()
```

**Step 2: Run tests to verify they pass**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/test_stt.py -v`
Expected: All 10 tests PASS.

**Step 3: Commit**

```bash
git add src/prot/stt.py tests/test_stt.py
git commit -m "feat(stt): migrate from Deepgram to ElevenLabs Scribe v2 Realtime

Replace Deepgram Nova-3 SDK with raw websockets + ElevenLabs realtime
STT protocol. Same public interface (connect/send_audio/disconnect +
callbacks) so pipeline.py needs zero changes.

- WebSocket JSON protocol with base64 audio chunks
- VAD commit strategy replaces utterance_end event
- partial_transcript → on_transcript(text, False)
- committed_transcript → on_transcript(text, True) + on_utterance_end()"
```

---

### Task 3: Update config — remove Deepgram settings, add STT language

**Files:**
- Modify: `src/prot/config.py:7,19-23`

**Step 1: Edit `config.py`**

Remove:
```python
    deepgram_api_key: str

    # Deepgram
    deepgram_model: str = "nova-3"
    deepgram_language: str = "ko"
    deepgram_endpointing: int = 500
    deepgram_utterance_end_ms: int = 2000
```

Add (in the Audio section, after `chunk_size`):
```python
    # STT
    stt_language: str = "ko"
```

The `elevenlabs_api_key` already exists in config (line 8), so no new API key field needed.

**Step 2: Run all tests**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/ -v`
Expected: All tests pass. If any test references `settings.deepgram_*`, it will fail and must be fixed.

**Step 3: Commit**

```bash
git add src/prot/config.py
git commit -m "refactor(config): remove Deepgram settings, add stt_language

- Remove deepgram_api_key, deepgram_model, deepgram_language,
  deepgram_endpointing, deepgram_utterance_end_ms
- Add stt_language (default: ko) used by ElevenLabs STT
- elevenlabs_api_key already exists for TTS, now shared with STT"
```

---

### Task 4: Clean up test fixtures and dependencies

**Files:**
- Modify: `tests/conftest.py:8` — remove `DEEPGRAM_API_KEY`
- Modify: `pyproject.toml:15` — remove `deepgram-sdk>=4.0`

**Step 1: Edit `conftest.py`**

Remove line 8:
```python
os.environ.setdefault("DEEPGRAM_API_KEY", "test-key")
```

**Step 2: Edit `pyproject.toml`**

Remove from dependencies:
```
    "deepgram-sdk>=4.0",
```

**Step 3: Run all tests**

Run: `cd /home/cyan/workplace/prot && python -m pytest tests/ -v`
Expected: All tests pass. No Deepgram imports remain.

**Step 4: Verify no Deepgram references remain**

Run: `grep -rn "deepgram\|Deepgram\|DEEPGRAM" src/ tests/ pyproject.toml`
Expected: Zero matches.

**Step 5: Commit**

```bash
git add tests/conftest.py pyproject.toml
git commit -m "chore: remove deepgram-sdk dependency and test fixtures

- Remove DEEPGRAM_API_KEY from conftest.py
- Remove deepgram-sdk from pyproject.toml dependencies
- Zero Deepgram references remain in codebase"
```

---

### Task 5: Remove .env DEEPGRAM_API_KEY and restart service

**Step 1: Edit `.env`**

Remove the `DEEPGRAM_API_KEY=...` line from `.env` file if present.

**Step 2: Sync dependencies**

Run: `cd /home/cyan/workplace/prot && uv sync`
This removes the `deepgram-sdk` package from the venv.

**Step 3: Restart and verify**

Run: `sudo systemctl restart prot`
Run: `journalctl -u prot -f --no-hostname` — watch for "Connected model=scribe_v2_realtime" log line.

**Step 4: Live test**

Speak Korean into the mic. Check journal logs for properly spaced Korean text in STT output.
If spacing works → migration success.
If spacing still broken → ElevenLabs also has this issue, need to evaluate other providers.

---

## Summary of changes

| File | Action | Lines changed |
|------|--------|---------------|
| `src/prot/stt.py` | Full rewrite | ~130 → ~130 |
| `src/prot/config.py` | Remove 6 lines, add 2 | -6, +2 |
| `tests/test_stt.py` | Full rewrite | ~134 → ~150 |
| `tests/conftest.py` | Remove 1 line | -1 |
| `pyproject.toml` | Remove 1 dependency | -1 |
| `.env` | Remove 1 key | -1 |
| `pipeline.py` | **No changes** | 0 |

**Zero interface changes** — `Pipeline` class continues to use the same `STTClient(on_transcript=..., on_utterance_end=...)` API.
