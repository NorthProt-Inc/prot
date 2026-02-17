"""FastAPI application with health endpoint and audio lifecycle."""

from __future__ import annotations

import os
import tracemalloc
from contextlib import asynccontextmanager

from fastapi import FastAPI

from prot.audio import AudioManager
from prot.config import settings
from prot.log import get_logger, setup_logging
from prot.pipeline import Pipeline

setup_logging()
logger = get_logger(__name__)

if os.environ.get("PROT_TRACEMALLOC"):
    tracemalloc.start()
    logger.info("tracemalloc enabled")

pipeline: Pipeline | None = None
audio: AudioManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, audio
    pipeline = Pipeline()
    await pipeline.startup()
    try:
        audio = AudioManager(
            device_index=settings.mic_device_index,
            sample_rate=settings.sample_rate,
            chunk_size=settings.chunk_size,
            on_audio=pipeline.on_audio_chunk,
        )
        audio.start()
    except Exception:
        logger.exception("Audio init failed — running headless")
        audio = None
    logger.info("prot started", mic=settings.mic_device_index)
    yield
    if audio:
        audio.stop()
    await pipeline.shutdown()
    logger.info("prot stopped")


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


@app.get("/diagnostics")
async def diagnostics():
    if not pipeline:
        return {"status": "not_started"}
    return pipeline.diagnostics()


@app.get("/memory")
async def memory_stats():
    if not tracemalloc.is_tracing():
        return {"error": "tracemalloc not enabled. Set PROT_TRACEMALLOC=1"}
    current, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    top = snapshot.statistics("lineno")[:20]
    return {
        "current_mb": round(current / 1024 / 1024, 2),
        "peak_mb": round(peak / 1024 / 1024, 2),
        "top_allocations": [
            {"file": str(s.traceback), "size_kb": round(s.size / 1024, 1)}
            for s in top
        ],
    }
