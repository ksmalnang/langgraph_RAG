"""OpenRouter embeddings client (OpenAI-compatible API)."""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenRouter's embeddings endpoint.

    Returns a list of embedding vectors (one per input text).
    """
    if not texts:
        return []

    settings = get_settings()
    url = f"{settings.openrouter_base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    embeddings: list[list[float]] = [
        item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
    ]
    logger.debug("Embedded %d text(s), dim=%d", len(texts), len(embeddings[0]))
    return embeddings


async def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    result = await embed_texts([text])
    return result[0]
