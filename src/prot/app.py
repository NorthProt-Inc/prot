"""FastAPI application with health endpoint and audio lifecycle."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from prot.audio import AudioManager
from prot.config import settings
from prot.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

pipeline: Pipeline | None = None
audio: AudioManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, audio
    pipeline = Pipeline()
    await pipeline.startup()
    audio = AudioManager(
        device_index=settings.mic_device_index,
        sample_rate=settings.sample_rate,
        chunk_size=settings.chunk_size,
        on_audio=pipeline.on_audio_chunk,
    )
    audio.start()
    print(f"prot started. Mic device={settings.mic_device_index}. Listening...")
    yield
    audio.stop()
    await pipeline.shutdown()
    print("prot stopped.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "state": pipeline.state.state.value if pipeline else "not_started",
    }


@app.get("/state")
async def state():
    return {"state": pipeline.state.state.value if pipeline else "not_started"}
