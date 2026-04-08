from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from qdrant_client import models

from app.services import vectorstore


@pytest.mark.asyncio
async def test_upsert_points_uses_bm25_named_sparse_vector():
    settings = SimpleNamespace(collection_name="admin_docs")
    mock_client = AsyncMock()

    with (
        patch("app.services.vectorstore.get_settings", return_value=settings),
        patch("app.services.vectorstore.get_qdrant_client", new=AsyncMock(return_value=mock_client)),
        patch(
            "app.services.vectorstore._encode_sparse",
            return_value=[models.SparseVector(indices=[1], values=[0.8])],
        ),
    ):
        await vectorstore.upsert_points(
            ids=["p1"],
            vectors=[[0.1, 0.2]],
            payloads=[{"text": "hello"}],
        )

    call_kwargs = mock_client.upsert.call_args.kwargs
    point = call_kwargs["points"][0]
    assert "bm25" in point.vector
    assert "sparse" not in point.vector


@pytest.mark.asyncio
async def test_hybrid_search_uses_bm25_prefetch_arm():
    settings = SimpleNamespace(
        collection_name="admin_docs",
        retrieval_top_k=5,
        hybrid_prefetch_limit=10,
    )
    mock_client = AsyncMock()
    mock_client.query_points.return_value = SimpleNamespace(points=[])

    async def passthrough_retry(*, fn, **_kwargs):
        return await fn()

    with (
        patch("app.services.vectorstore.get_settings", return_value=settings),
        patch("app.services.vectorstore.get_qdrant_client", new=AsyncMock(return_value=mock_client)),
        patch(
            "app.services.vectorstore._encode_sparse_query",
            return_value=models.SparseVector(indices=[1], values=[0.8]),
        ),
        patch("app.services.vectorstore.retry_async", new=passthrough_retry),
    ):
        await vectorstore.hybrid_search(
            query_text="test query",
            query_vector=[0.1, 0.2],
        )

    prefetch = mock_client.query_points.call_args.kwargs["prefetch"]
    assert prefetch[0].using == "bm25"

