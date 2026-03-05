"""Application configuration loaded from environment variables."""

from functools import lru_cache

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
    relevance_threshold: float = 0.20
    max_rewrite_count: int = 2
    filter_negative_scores: bool = True

    # ── Chunking ────────────────────────────────────────
    chunk_max_tokens: int = 512

    # ── App ─────────────────────────────────────────────
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()  # type: ignore[call-arg]
