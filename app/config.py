"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — values come from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenRouter ──────────────────────────────────────
    openrouter_api_key: str
    embedding_model: str = "qwen/qwen3-embedding-8b"
    llm_model: str = "deepseek/deepseek-v3.2"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Jina ────────────────────────────────────────────
    jina_api_key: str
    jina_reranker_model: str = "jina-reranker-v3"

    # ── Qdrant ──────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "admin_docs"

    # ── Redis ───────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 3600  # 1 hour

    # ── Retrieval tuning ────────────────────────────────
    retrieval_top_k: int = 10
    rerank_top_n: int = 5
    hybrid_prefetch_limit: int = 20
    relevance_threshold: float = 0.17
    max_rewrite_count: int = 2
    filter_negative_scores: bool = True

    # ── Chunking ────────────────────────────────────────
    chunk_max_tokens: int = 512

    # ── App ─────────────────────────────────────────────
    log_level: str = "INFO"
    app_env: str = "development"
    cors_allow_origins: str = ""
    ingest_api_key: str | None = None

    # ── Route abuse protection ──────────────────────────
    rate_limit_window_seconds: int = 60
    auth_login_rate_limit: int = 8
    chat_rate_limit: int = 40
    ingest_rate_limit: int = 6

    # ── Integration hardening ───────────────────────────
    llm_timeout_seconds: float = 45.0
    embedding_timeout_seconds: float = 30.0
    reranker_timeout_seconds: float = 20.0
    qdrant_timeout_seconds: float = 20.0
    redis_socket_timeout_seconds: float = 5.0
    siakad_timeout_seconds: float = 30.0

    # Retries are only used for idempotent remote reads.
    service_retry_attempts: int = 2
    service_retry_backoff_seconds: float = 0.3

    @field_validator("app_env")
    @classmethod
    def normalize_app_env(cls, value: str) -> str:
        return value.strip().lower()


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()  # type: ignore[call-arg]
