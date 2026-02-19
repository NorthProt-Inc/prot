from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings

LOCAL_TZ = ZoneInfo("America/Vancouver")


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str
    elevenlabs_api_key: str

    # Audio
    mic_device_index: int | None = None
    sample_rate: int = 16000
    chunk_size: int = 512

    # VAD
    vad_threshold: float = 0.5
    vad_threshold_speaking: float = 0.8
    vad_prebuffer_chunks: int = 8

    # STT
    stt_language: str = "ko"

    # LLM
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 4096
    claude_effort: str = "high"
    context_max_turns: int = 10

    # TTS
    elevenlabs_model: str = "eleven_v3"
    elevenlabs_voice_id: str = "s3lKyrFAzTUpzy3ZLwbM"
    elevenlabs_output_format: str = "pcm_24000"

    # HASS
    hass_url: str = "http://localhost:8123"
    hass_token: str = ""

    # Database
    database_url: str = "postgresql://prot:prot@localhost:5432/prot"
    db_pool_min: int = 2
    db_pool_max: int = 10
    db_export_dir: str = "data/db"

    # Embeddings
    voyage_api_key: str = ""
    voyage_model: str = "voyage-4"
    voyage_context_model: str = "voyage-context-3"

    # Reranker
    rerank_model: str = "rerank-2.5"
    rerank_top_k: int = 5

    # Memory
    memory_extraction_model: str = "claude-sonnet-4-6"
    memory_extraction_window_turns: int = 3
    rag_context_target_tokens: int = 4096
    rag_top_k: int = 10

    # Community detection
    community_rebuild_interval: int = 5
    community_min_entities: int = 5

    # Logging
    log_level: str = "INFO"

    # Timers
    active_timeout: int = 30  # seconds before WS disconnect

    model_config = {"env_file": ".env"}


settings = Settings()
