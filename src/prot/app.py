"""FastAPI application with health endpoint and audio lifecycle."""

from __future__ import annotations

import os
import tracemalloc
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from prot.audio import AudioManager
from prot.config import settings
from prot.context import ContextManager
from prot.engine import BusyError, ConversationEngine, ToolIterationMarker
from prot.persona import load_persona
from prot.logging import get_logger, setup_logging
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "state": pipeline.current_state if pipeline else "not_started",
    }


@app.get("/state")
async def state():
    return {"state": pipeline.current_state if pipeline else "not_started"}


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


@app.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    """Text chat with per-connection ConversationEngine."""
    await websocket.accept()
    if not pipeline:
        await websocket.send_json({"type": "error", "message": "Server not ready"})
        await websocket.close()
        return

    # Chat gets its own context with channel="chat"
    ctx = ContextManager(persona_text=load_persona(), channel="chat")
    engine = ConversationEngine(
        ctx=ctx,
        llm=pipeline._llm,
        hass_agent=pipeline._hass_agent,
        memory=pipeline._memory,
        graphrag=pipeline._graphrag,
    )
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") != "message" or not data.get("content"):
                continue

            engine.add_user_message(data["content"])

            try:
                async for item in engine.respond():
                    if isinstance(item, ToolIterationMarker):
                        continue
                    await websocket.send_json({"type": "chunk", "content": item})

                result = engine.last_result
                await websocket.send_json(
                    {"type": "done", "full_text": result.full_text if result else ""}
                )
            except BusyError:
                await websocket.send_json(
                    {"type": "error", "message": "Still processing previous message"}
                )
            except Exception as exc:
                logger.exception("Chat respond error")
                await websocket.send_json(
                    {"type": "error", "message": str(exc)}
                )
    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception:
        logger.exception("Chat WebSocket error")
    finally:
        # Extract memories from chat session before cleanup
        await engine.shutdown_summarize()
