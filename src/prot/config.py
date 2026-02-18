from pydantic_settings import BaseSettings


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

    # STT
    stt_language: str = "ko"

    # LLM
    claude_model: str = "claude-opus-4-6"
    claude_max_tokens: int = 1500
    claude_effort: str = "medium"

    # TTS
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_voice_id: str = "Fahco4VZzobUeiPqni1S"
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
    voyage_model: str = "voyage-4-lite"
    voyage_dimension: int = 1024

    # Memory
    memory_extraction_model: str = "claude-haiku-4-5-20251001"
    rag_context_target_tokens: int = 3000
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
