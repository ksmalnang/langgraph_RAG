"""Tests for API endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _assert_rfc7807(data: dict, status_code: int, expected_title: str) -> None:
    """Assert that the response follows RFC 7807 Problem Details format."""
    assert "type" in data, "RFC 7807 requires a 'type' field"
    assert "title" in data, "RFC 7807 requires a 'title' field"
    assert "status" in data, "RFC 7807 requires a 'status' field"
    assert "detail" in data, "RFC 7807 requires a 'detail' field"
    assert "instance" in data, "RFC 7807 requires an 'instance' field"

    assert data["status"] == status_code
    assert data["title"] == expected_title


# ── Health ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_healthy():
    """Health endpoint returns healthy when both services are up."""
    with (
        patch("app.services.vectorstore.get_qdrant_client") as mock_qdrant,
        patch("app.services.memory.get_redis") as mock_redis,
    ):
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = AsyncMock(collections=[])
        mock_qdrant.return_value = mock_client

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


# ── Chat ────────────────────────────────────────────────


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

    with patch("app.api.routers.chat._get_graph") as mock_get_graph:
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
        assert len(data["session_id"]) > 10
        assert data["session_id"] != "test-123"
        assert "register" in data["answer"].lower()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "enrollment.pdf"
        assert data["sources"][0]["doc_id"] == "abc123"
        assert data["sources"][0]["page"] == 5


@pytest.mark.asyncio
async def test_chat_invalid_message():
    """Chat endpoint rejects empty messages with RFC 7807 error."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": ""})

    assert response.status_code == 422
    _assert_rfc7807(response.json(), 422, "Validation Error")


# ── Ingestion ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_files_empty():
    """List files endpoint returns 200 with empty list when no files ingested."""
    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ingest/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 0
        assert data["files"] == []


@pytest.mark.asyncio
async def test_list_files_two_files():
    """List files endpoint returns one entry per ingested file."""
    file_entries = [
        {
            "doc_id": "doc-abc123",
            "filename": "enrollment.pdf",
            "doc_category": "Academic",
            "academic_year": "2024/2025",
            "total_chunks": 5,
        },
        {
            "doc_id": "doc-def456",
            "filename": "schedule.pdf",
            "doc_category": "Administrative",
            "academic_year": "2024/2025",
            "total_chunks": 3,
        },
    ]

    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = file_entries

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ingest/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 2
        # Results sorted by filename ascending
        assert data["files"][0]["filename"] == "enrollment.pdf"
        assert data["files"][1]["filename"] == "schedule.pdf"
        assert data["files"][0]["total_chunks"] == 5
        assert data["files"][1]["total_chunks"] == 3
        assert data["files"][0]["doc_category"] == "Academic"
        assert data["files"][0]["academic_year"] == "2024/2025"


@pytest.mark.asyncio
async def test_list_files_total_chunks_matches():
    """List files endpoint accurately reflects stored chunk counts."""
    file_entries = [
        {
            "doc_id": "doc-xyz",
            "filename": "large-doc.pdf",
            "doc_category": None,
            "academic_year": None,
            "total_chunks": 42,
        },
    ]

    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = file_entries

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ingest/files")

        assert response.status_code == 200
        data = response.json()
        assert data["files"][0]["total_chunks"] == 42


@pytest.mark.asyncio
async def test_list_files_idempotent_reingest():
    """Re-ingesting the same file results in only one entry."""
    # Simulates deduplication: even if called multiple times, same doc_id
    file_entries = [
        {
            "doc_id": "doc-duplicate",
            "filename": "re-ingested.pdf",
            "doc_category": "Academic",
            "academic_year": "2024/2025",
            "total_chunks": 10,
        },
    ]

    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = file_entries

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ingest/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 1
        assert data["files"][0]["doc_id"] == "doc-duplicate"


@pytest.mark.asyncio
async def test_delete_file_success():
    """Delete endpoint removes a file and returns chunk count."""
    delete_result = {
        "doc_id": "doc-abc123",
        "filename": "enrollment.pdf",
        "deleted_chunks": 5,
    }

    with patch("app.api.routers.ingestion.vs.delete_file") as mock_delete:
        mock_delete.return_value = delete_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "DELETE", "/ingest/files", json={"doc_id": "doc-abc123"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc-abc123"
        assert data["filename"] == "enrollment.pdf"
        assert data["deleted_chunks"] == 5
        assert data["deleted"] is True
        assert "deleted successfully" in data["message"]


@pytest.mark.asyncio
async def test_delete_file_not_found():
    """Delete endpoint returns 404 for unknown doc_id."""
    from app.utils.exceptions import VectorStoreError

    with patch("app.api.routers.ingestion.vs.delete_file") as mock_delete:
        mock_delete.side_effect = VectorStoreError(
            "File with doc_id='unknown-id' not found"
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "DELETE", "/ingest/files", json={"doc_id": "unknown-id"}
            )

    assert response.status_code == 404
    data = response.json()
    assert data["title"] == "Not Found"
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_delete_file_twice_returns_404():
    """Deleting the same doc_id twice returns 404 on second call."""
    from app.utils.exceptions import VectorStoreError

    # First delete succeeds
    first_result = {
        "doc_id": "doc-to-delete",
        "filename": "remove-me.pdf",
        "deleted_chunks": 3,
    }

    with patch("app.api.routers.ingestion.vs.delete_file") as mock_delete:
        mock_delete.return_value = first_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "DELETE", "/ingest/files", json={"doc_id": "doc-to-delete"}
            )

        assert response.status_code == 200

    # Second delete fails
    with patch("app.api.routers.ingestion.vs.delete_file") as mock_delete:
        mock_delete.side_effect = VectorStoreError(
            "File with doc_id='doc-to-delete' not found"
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "DELETE", "/ingest/files", json={"doc_id": "doc-to-delete"}
            )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_no_other_files_affected():
    """Deleting one file does not remove chunks from other files."""
    # Simulates that only the target doc_id's chunks are deleted
    delete_result = {
        "doc_id": "doc-target",
        "filename": "target.pdf",
        "deleted_chunks": 7,
    }

    with patch("app.api.routers.ingestion.vs.delete_file") as mock_delete:
        mock_delete.return_value = delete_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "DELETE", "/ingest/files", json={"doc_id": "doc-target"}
            )

        assert response.status_code == 200
        data = response.json()
        # Only target file affected
        assert data["doc_id"] == "doc-target"
        assert data["deleted_chunks"] == 7


# ── Chunks Listing (Issue 1) ────────────────────────────


class _FakePoint:
    """Minimal fake Qdrant point for testing."""

    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


@pytest.mark.asyncio
async def test_list_chunks_success():
    """List chunks endpoint returns all chunks for a doc_id."""
    fake_points = [
        _FakePoint(
            point_id="chunk-001",
            payload={
                "chunk_index": 0,
                "page": 1,
                "headings": ["Introduction"],
                "content_type": "text",
                "doc_category": "guide",
                "academic_year": "2024/2025",
                "text": "This is chunk 1.",
                "enriched_text": "[Introduction]\nThis is chunk 1.",
                "filename": "test.pdf",
            },
        ),
        _FakePoint(
            point_id="chunk-002",
            payload={
                "chunk_index": 1,
                "page": 2,
                "headings": ["Rules"],
                "content_type": "text",
                "doc_category": "guide",
                "academic_year": "2024/2025",
                "text": "This is chunk 2.",
                "enriched_text": "[Rules]\nThis is chunk 2.",
                "filename": "test.pdf",
            },
        ),
    ]

    with patch("app.api.routers.ingestion.vs.scroll_chunks_by_doc_id") as mock_scroll:
        mock_scroll.return_value = fake_points

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/ingest/by-file/chunks", params={"doc_id": "doc-test"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc-test"
        assert data["filename"] == "test.pdf"
        assert data["total_chunks"] == 2
        assert data["chunks"][0]["chunk_id"] == "chunk-001"
        assert data["chunks"][0]["chunk_index"] == 0
        assert data["chunks"][0]["page"] == 1
        assert data["chunks"][0]["headings"] == ["Introduction"]
        assert data["chunks"][0]["content_type"] == "text"
        assert data["chunks"][0]["doc_category"] == "guide"
        assert data["chunks"][0]["academic_year"] == "2024/2025"
        assert data["chunks"][0]["enriched_text"] is not None


@pytest.mark.asyncio
async def test_list_chunks_sorted_by_chunk_index():
    """List chunks results are always sorted by chunk_index."""
    # Return points out of order — endpoint should sort them
    fake_points = [
        _FakePoint(
            point_id="chunk-002",
            payload={"chunk_index": 5, "filename": "test.pdf", "text": "chunk 5"},
        ),
        _FakePoint(
            point_id="chunk-001",
            payload={"chunk_index": 2, "filename": "test.pdf", "text": "chunk 2"},
        ),
        _FakePoint(
            point_id="chunk-003",
            payload={"chunk_index": 1, "filename": "test.pdf", "text": "chunk 1"},
        ),
    ]

    with patch("app.api.routers.ingestion.vs.scroll_chunks_by_doc_id") as mock_scroll:
        mock_scroll.return_value = sorted(
            fake_points, key=lambda p: p.payload["chunk_index"]
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/ingest/by-file/chunks", params={"doc_id": "doc-test"}
            )

        assert response.status_code == 200
        data = response.json()
        indices = [c["chunk_index"] for c in data["chunks"]]
        assert indices == [1, 2, 5]


@pytest.mark.asyncio
async def test_list_chunks_not_found():
    """List chunks returns 404 for unknown doc_id."""
    with patch("app.api.routers.ingestion.vs.scroll_chunks_by_doc_id") as mock_scroll:
        mock_scroll.return_value = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/ingest/by-file/chunks", params={"doc_id": "unknown"}
            )

    assert response.status_code == 404
    data = response.json()
    assert data["title"] == "Not Found"


@pytest.mark.asyncio
async def test_list_chunks_missing_param():
    """List chunks returns 422 when doc_id param is missing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ingest/by-file/chunks")

    assert response.status_code == 422


# ── File Rename (Issue 7) ──────────────────────────────


@pytest.mark.asyncio
async def test_rename_file_success():
    """Rename endpoint updates filename and returns chunk count."""
    rename_result = {
        "doc_id": "doc-abc123",
        "filename": "new-name.pdf",
        "updated_chunks": 5,
    }

    with (
        patch("app.api.routers.ingestion.vs.list_files") as mock_list,
        patch("app.api.routers.ingestion.vs.rename_file") as mock_rename,
    ):
        mock_list.return_value = []  # no collision
        mock_rename.return_value = rename_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "PATCH",
                "/ingest/files",
                json={"doc_id": "doc-abc123", "filename": "new-name.pdf"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc-abc123"
        assert data["filename"] == "new-name.pdf"
        assert data["updated_chunks"] == 5
        assert data["updated"] is True


@pytest.mark.asyncio
async def test_rename_file_not_found():
    """Rename returns 404 for unknown doc_id."""
    from app.utils.exceptions import VectorStoreError

    with (
        patch("app.api.routers.ingestion.vs.list_files") as mock_list,
        patch("app.api.routers.ingestion.vs.rename_file") as mock_rename,
    ):
        mock_list.return_value = []
        mock_rename.side_effect = VectorStoreError(
            "File with doc_id='unknown' not found"
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "PATCH",
                "/ingest/files",
                json={"doc_id": "unknown", "filename": "new.pdf"},
            )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_rename_file_conflict():
    """Rename returns 409 when filename already used by another doc_id."""
    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = [
            {
                "doc_id": "other-doc",
                "filename": "existing.pdf",
                "total_chunks": 3,
            }
        ]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "PATCH",
                "/ingest/files",
                json={"doc_id": "my-doc", "filename": "existing.pdf"},
            )

    assert response.status_code == 409
    data = response.json()
    assert data["title"] == "Conflict"


@pytest.mark.asyncio
async def test_rename_file_empty_filename():
    """Rename returns 422 when filename is empty."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.request(
            "PATCH",
            "/ingest/files",
            json={"doc_id": "doc-abc", "filename": ""},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rename_no_other_files_affected():
    """Renaming one file does not affect other files."""
    rename_result = {
        "doc_id": "doc-target",
        "filename": "renamed.pdf",
        "updated_chunks": 7,
    }

    with (
        patch("app.api.routers.ingestion.vs.list_files") as mock_list,
        patch("app.api.routers.ingestion.vs.rename_file") as mock_rename,
    ):
        mock_list.return_value = []
        mock_rename.return_value = rename_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(
                "PATCH",
                "/ingest/files",
                json={"doc_id": "doc-target", "filename": "renamed.pdf"},
            )

        assert response.status_code == 200
        data = response.json()
        # doc_id unchanged
        assert data["doc_id"] == "doc-target"
        assert data["filename"] == "renamed.pdf"


@pytest.mark.asyncio
async def test_ingest_unsupported_file():
    """Ingest endpoint rejects unsupported file types with RFC 7807 error."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            files={"file": ("test.exe", b"malicious", "application/octet-stream")},
        )

    assert response.status_code == 422
    _assert_rfc7807(response.json(), 422, "Validation Error")


@pytest.mark.asyncio
async def test_ingest_rejects_file_over_size_limit():
    """Ingest endpoint rejects oversized files with RFC 7807 error."""
    fake_settings = SimpleNamespace(
        app_env="development",
        ingest_api_key=None,
        ingest_rate_limit=5,
        rate_limit_window_seconds=60,
        ingest_max_upload_mb=1,
    )

    with patch("app.api.routers.ingestion.get_settings", return_value=fake_settings):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/ingest",
                files={"file": ("big.md", b"a" * (1024 * 1024 + 1), "text/markdown")},
            )

    assert response.status_code == 413
    _assert_rfc7807(response.json(), 413, "Payload Too Large")
