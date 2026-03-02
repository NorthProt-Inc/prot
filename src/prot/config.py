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
    barge_in_enabled: bool = False

    # STT
    stt_language: str = "ko"
    stt_silence_threshold_secs: float = 3.0

    # LLM
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 4096
    claude_effort: str = "high"
    compaction_trigger: int = 50000
    tool_clear_trigger: int = 30000
    tool_clear_keep: int = 3
    thinking_keep_turns: int = 2

    # TTS
    elevenlabs_model: str = "eleven_v3"
    elevenlabs_voice_id: str = "s3lKyrFAzTUpzy3ZLwbM"
    elevenlabs_output_format: str = "pcm_24000"
    tts_sentence_silence_ms: int = 200

    # HASS
    hass_url: str = "http://localhost:8123"
    hass_token: str = ""
    hass_agent_id: str = "conversation.google_ai_conversation"

    # Database
    database_url: str = "postgresql://prot:prot@localhost:5432/prot"
    db_pool_min: int = 2
    db_pool_max: int = 10

    # Embeddings
    voyage_api_key: str = ""
    voyage_model: str = "voyage-4-large"

    # Reranker
    rerank_model: str = "rerank-2.5"
    rerank_top_k: int = 5

    # Memory
    memory_extraction_model: str = "claude-haiku-4-5-20251001"
    rag_context_target_tokens: int = 4096
    rag_top_k: int = 10

    # Compaction
    pause_after_compaction: bool = True

    # Decay
    decay_base_rate: float = 0.002
    decay_min_retention: float = 0.1

    # Logging
    log_level: str = "INFO"

    # Timers
    active_timeout: int = 30  # seconds before WS disconnect

    model_config = {"env_file": ".env"}


settings = Settings()
