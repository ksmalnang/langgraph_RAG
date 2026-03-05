"""Tests for API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_health_healthy():
    """Health endpoint returns healthy when both services are up."""
    with (
        patch("app.services.vectorstore.get_qdrant_client") as mock_qdrant,
        patch("app.services.memory.get_redis") as mock_redis,
    ):
        # Mock Qdrant
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = AsyncMock(collections=[])
        mock_qdrant.return_value = mock_client

        # Mock Redis
        mock_r = AsyncMock()
        mock_r.ping.return_value = True
        mock_redis.return_value = mock_r

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["qdrant"] == "connected"
        assert data["redis"] == "connected"


@pytest.mark.asyncio
async def test_health_degraded_redis():
    """Health endpoint returns degraded when Redis is down."""
    with (
        patch("app.services.vectorstore.get_qdrant_client") as mock_qdrant,
        patch("app.services.memory.get_redis") as mock_redis,
    ):
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = AsyncMock(collections=[])
        mock_qdrant.return_value = mock_client

        mock_redis.side_effect = ConnectionError("Redis down")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["redis"] == "disconnected"


@pytest.mark.asyncio
async def test_chat_endpoint():
    """Chat endpoint invokes the graph and returns a response."""
    mock_result = {
        "answer": "You can register through the portal.",
        "sources": [
            {
                "doc_id": "abc123",
                "filename": "enrollment.pdf",
                "page": 5,
                "score": 0.95,
                "snippet": "To register for next semester...",
            }
        ],
    }

    with patch("app.api.chat._get_graph") as mock_get_graph:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat",
                json={"session_id": "test-123", "message": "How do I register?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-123"
        assert "register" in data["answer"].lower()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "enrollment.pdf"
        assert data["sources"][0]["doc_id"] == "abc123"
        assert data["sources"][0]["page"] == 5


@pytest.mark.asyncio
async def test_chat_missing_fields():
    """Chat endpoint rejects requests with missing fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ingest_unsupported_file():
    """Ingest endpoint rejects unsupported file types."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            files={"file": ("test.exe", b"malicious", "application/octet-stream")},
        )

    assert response.status_code == 422
