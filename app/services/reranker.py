"""Jina Reranker v3 API client."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.services.resilience import elapsed_ms, now_ms, retry_async
from app.utils.exceptions import RerankerError
from app.utils.logger import get_logger

logger = get_logger(__name__)

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


async def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank documents using Jina Reranker v3.

    Each document dict must contain a ``"text"`` key.
    Returns a list of documents with an added ``"relevance_score"`` key,
    sorted by descending relevance.
    """
    settings = get_settings()
    top_n = top_n or settings.rerank_top_n

    if not documents:
        return []

    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": settings.jina_reranker_model,
        "query": query,
        "documents": [doc["text"] for doc in documents],
        "top_n": top_n,
    }

    start_ms = now_ms()

    async def _request() -> dict:
        async with httpx.AsyncClient(timeout=settings.reranker_timeout_seconds) as client:
            response = await client.post(JINA_RERANK_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    try:
        data = await retry_async(
            operation="rerank",
            dependency="jina",
            fn=_request,
            retry_on=(httpx.TimeoutException, httpx.TransportError),
        )
    except httpx.TimeoutException as exc:
        logger.error(
            "Dependency failure dependency=jina operation=rerank mode=timeout latency_ms=%.1f",
            elapsed_ms(start_ms),
        )
        raise RerankerError("Reranker request timed out") from exc
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Dependency failure dependency=jina operation=rerank mode=http_status status=%s latency_ms=%.1f",
            exc.response.status_code,
            elapsed_ms(start_ms),
        )
        raise RerankerError("Reranker request failed with upstream HTTP error") from exc
    except httpx.TransportError as exc:
        logger.error(
            "Dependency failure dependency=jina operation=rerank mode=transport latency_ms=%.1f",
            elapsed_ms(start_ms),
        )
        raise RerankerError("Reranker request failed due to network error") from exc
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=jina operation=rerank mode=unexpected latency_ms=%.1f",
            elapsed_ms(start_ms),
            exc_info=True,
        )
        raise RerankerError("Reranker request failed unexpectedly") from exc

    reranked: list[dict[str, Any]] = []
    for item in data.get("results", []):
        idx = item["index"]
        doc = {**documents[idx]}
        doc["relevance_score"] = item["relevance_score"]
        reranked.append(doc)

    # Sort descending by relevance
    reranked.sort(key=lambda d: d["relevance_score"], reverse=True)
    logger.debug(
        "Reranked %d → %d docs, top score=%.4f",
        len(documents),
        len(reranked),
        reranked[0]["relevance_score"] if reranked else 0.0,
    )
    return reranked
