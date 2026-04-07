from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import uuid

from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app
from app.services.rate_limiter import reset_rate_limiter


@pytest.fixture(autouse=True)
def _reset_limiter():
    reset_rate_limiter()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_chat_requires_student_access_token_for_student_bound_session():
    valid_session_id = str(uuid.uuid4())

    with (
        patch("app.api.chat.has_student_access_binding", new_callable=AsyncMock) as mock_has,
        patch(
            "app.api.chat.verify_student_access_token",
            new_callable=AsyncMock,
        ) as mock_verify,
    ):
        mock_has.return_value = True
        mock_verify.return_value = False

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={"session_id": valid_session_id, "message": "cek nilai saya"},
            )

    assert resp.status_code == 401
    assert "token" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_chat_allows_student_bound_session_with_valid_access_token():
    valid_session_id = str(uuid.uuid4())
    mock_result = {"answer": "ok", "sources": []}

    with (
        patch("app.api.chat.has_student_access_binding", new_callable=AsyncMock) as mock_has,
        patch(
            "app.api.chat.verify_student_access_token",
            new_callable=AsyncMock,
        ) as mock_verify,
        patch("app.api.chat._get_graph") as mock_get_graph,
    ):
        mock_has.return_value = True
        mock_verify.return_value = True
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={"session_id": valid_session_id, "message": "cek nilai saya"},
                headers={"X-Student-Access-Token": "valid"},
            )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "ok"


@pytest.mark.asyncio
async def test_ingest_disabled_without_key_in_non_local_env():
    fake_settings = SimpleNamespace(
        app_env="production",
        ingest_api_key=None,
        ingest_rate_limit=5,
        rate_limit_window_seconds=60,
    )

    with patch("app.api.ingest.get_settings", return_value=fake_settings):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ingest",
                files={"file": ("a.md", b"# test", "text/markdown")},
            )

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_ingest_requires_token_when_configured():
    fake_settings = SimpleNamespace(
        app_env="production",
        ingest_api_key="top-secret",
        ingest_rate_limit=5,
        rate_limit_window_seconds=60,
    )

    with patch("app.api.ingest.get_settings", return_value=fake_settings):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ingest",
                files={"file": ("a.md", b"# test", "text/markdown")},
            )

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_chat_rate_limited():
    fake_settings = SimpleNamespace(
        chat_rate_limit=1,
        rate_limit_window_seconds=60,
    )
    mock_result = {"answer": "ok", "sources": []}

    with (
        patch("app.api.chat.get_settings", return_value=fake_settings),
        patch("app.api.chat._get_graph") as mock_get_graph,
    ):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/chat", json={"message": "hi"})
            second = await client.post("/chat", json={"message": "again"})

    assert first.status_code == 200
    assert second.status_code == 429

