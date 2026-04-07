"""OpenRouter embeddings client (OpenAI-compatible API)."""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.services.resilience import elapsed_ms, now_ms, retry_async
from app.utils.exceptions import EmbeddingError
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

    start_ms = now_ms()

    async def _request() -> dict:
        async with httpx.AsyncClient(timeout=settings.embedding_timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    try:
        data = await retry_async(
            operation="embed_texts",
            dependency="openrouter",
            fn=_request,
            retry_on=(httpx.TimeoutException, httpx.TransportError),
        )
    except httpx.TimeoutException as exc:
        logger.error(
            "Dependency failure dependency=openrouter operation=embed_texts mode=timeout latency_ms=%.1f",
            elapsed_ms(start_ms),
        )
        raise EmbeddingError("Embedding request timed out") from exc
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Dependency failure dependency=openrouter operation=embed_texts mode=http_status status=%s latency_ms=%.1f",
            exc.response.status_code,
            elapsed_ms(start_ms),
        )
        raise EmbeddingError("Embedding request failed with upstream HTTP error") from exc
    except httpx.TransportError as exc:
        logger.error(
            "Dependency failure dependency=openrouter operation=embed_texts mode=transport latency_ms=%.1f",
            elapsed_ms(start_ms),
        )
        raise EmbeddingError("Embedding request failed due to network error") from exc
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=openrouter operation=embed_texts mode=unexpected latency_ms=%.1f",
            elapsed_ms(start_ms),
            exc_info=True,
        )
        raise EmbeddingError("Embedding request failed unexpectedly") from exc

    embeddings: list[list[float]] = [
        item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
    ]
    logger.debug("Embedded %d text(s), dim=%d", len(texts), len(embeddings[0]))
    return embeddings


async def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    result = await embed_texts([text])
    return result[0]
