from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.utils.exceptions import EmbeddingError, MemoryStoreError, RerankerError


@pytest.mark.asyncio
async def test_embed_texts_timeout_maps_to_embedding_error():
    from app.services.embeddings import embed_texts

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.TimeoutException("timed out")

    with patch("app.services.embeddings.httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(EmbeddingError, match="timed out"):
            await embed_texts(["hello"])


@pytest.mark.asyncio
async def test_rerank_http_status_maps_to_reranker_error():
    from app.services.reranker import rerank

    request = httpx.Request("POST", "https://api.jina.ai/v1/rerank")
    response = httpx.Response(status_code=503, request=request)

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "upstream error",
        request=request,
        response=response,
    )

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    with patch("app.services.reranker.httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(RerankerError, match="HTTP error"):
            await rerank("query", [{"text": "doc"}], top_n=1)


@pytest.mark.asyncio
async def test_memory_get_history_maps_to_memorystore_error():
    from app.services.memory import get_history

    with patch("app.services.memory.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RuntimeError("redis down")
        with pytest.raises(MemoryStoreError, match="load chat history"):
            await get_history("session-1")

