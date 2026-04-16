"""Tests for Telegram Bot webhook integration.

Covers:
  - Pydantic model parsing
  - Session ID generation
  - Webhook endpoint: token rejection, dedup, non-text drop, valid processing
  - Setup / delete webhook endpoints
  - Background handler: success path, error fallback
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient
import pytest

from app.api.models import TelegramChat, TelegramMessage, TelegramUpdate, WebhookSetupRequest
from app.main import app
from app.services.telegram_handler import build_telegram_session_id, process_telegram_message_background


# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------

_VALID_TOKEN = "1234567890:AAABBBCCC"  # fake token used across tests

_FAKE_SETTINGS = SimpleNamespace(telegram_bot_token=_VALID_TOKEN)


def _update(
    update_id: int = 1,
    chat_id: int = 999,
    text: str | None = "Hello bot!",
) -> dict:
    """Build a minimal Telegram Update JSON payload."""
    return {
        "update_id": update_id,
        "message": {
            "message_id": 42,
            "date": 1_700_000_000,
            "chat": {"id": chat_id, "type": "private", "first_name": "Test"},
            "text": text,
        },
    }


# ---------------------------------------------------------------------------
# Unit: Pydantic models
# ---------------------------------------------------------------------------


class TestTelegramModels:
    def test_telegram_chat_required_fields(self):
        chat = TelegramChat(id=123456789, type="private")
        assert chat.id == 123456789
        assert chat.type == "private"
        assert chat.first_name is None

    def test_telegram_message_text_optional(self):
        msg = TelegramMessage(
            message_id=1,
            date=1_700_000_000,
            chat=TelegramChat(id=1, type="private"),
        )
        assert msg.text is None  # non-text; sticker / photo etc.

    def test_telegram_update_no_message(self):
        update = TelegramUpdate(update_id=99)
        assert update.message is None

    def test_webhook_setup_request_url(self):
        req = WebhookSetupRequest(url="https://abc.ngrok.io")
        assert req.url == "https://abc.ngrok.io"


# ---------------------------------------------------------------------------
# Unit: session ID builder
# ---------------------------------------------------------------------------


class TestBuildTelegramSessionId:
    def test_format(self):
        assert build_telegram_session_id(123456789) == "tg_123456789"

    def test_large_int64(self):
        """Telegram chat IDs can be large (52 bits active)."""
        large_id = 2**40
        result = build_telegram_session_id(large_id)
        assert result == f"tg_{large_id}"
        assert result.startswith("tg_")

    def test_negative_ids_for_groups(self):
        """Group chat IDs are negative in Telegram."""
        assert build_telegram_session_id(-100987654321) == "tg_-100987654321"

    def test_distinct_from_uuid_format(self):
        """Session IDs must NOT be UUID-like to avoid confusion with HTTP chat sessions."""
        session = build_telegram_session_id(42)
        assert "-" not in session  # no UUID hyphens
        assert session.startswith("tg_")


# ---------------------------------------------------------------------------
# Integration: webhook endpoint
# ---------------------------------------------------------------------------


class TestTelegramWebhookEndpoint:
    @pytest.mark.asyncio
    async def test_valid_update_returns_200(self):
        """A valid text message is accepted and returns 200 immediately."""
        with (
            patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.api.routers.telegram._is_duplicate_update", return_value=False),
            patch("app.api.routers.telegram.process_telegram_message_background") as mock_bg,
        ):
            mock_bg.return_value = None  # background task, not awaited directly

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    f"/telegram/webhook/{_VALID_TOKEN}",
                    json=_update(),
                )

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_invalid_token_returns_200_silently(self):
        """Invalid token is silently acknowledged (returns 200) to prevent Telegram retries."""
        with patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/telegram/webhook/WRONG_TOKEN",
                    json=_update(),
                )

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_duplicate_update_is_dropped(self):
        """Duplicate update_id is acknowledged but not processed."""
        with (
            patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.api.routers.telegram._is_duplicate_update", return_value=True),
            patch("app.api.routers.telegram.process_telegram_message_background") as mock_bg,
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    f"/telegram/webhook/{_VALID_TOKEN}",
                    json=_update(update_id=7),
                )

        assert response.status_code == 200
        mock_bg.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_text_update_is_dropped(self):
        """Sticker / photo updates (text=None) are acknowledged and not processed."""
        with (
            patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.api.routers.telegram._is_duplicate_update", return_value=False),
            patch("app.api.routers.telegram.process_telegram_message_background") as mock_bg,
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    f"/telegram/webhook/{_VALID_TOKEN}",
                    json=_update(text=None),  # sticker / photo
                )

        assert response.status_code == 200
        mock_bg.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_without_message_is_dropped(self):
        """Update with no message field is silently acknowledged."""
        with (
            patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.api.routers.telegram._is_duplicate_update", return_value=False),
            patch("app.api.routers.telegram.process_telegram_message_background") as mock_bg,
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    f"/telegram/webhook/{_VALID_TOKEN}",
                    json={"update_id": 5},  # no message
                )

        assert response.status_code == 200
        mock_bg.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_text_is_dropped(self):
        """A message containing only whitespace is not forwarded to the LLM."""
        with (
            patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.api.routers.telegram._is_duplicate_update", return_value=False),
            patch("app.api.routers.telegram.process_telegram_message_background") as mock_bg,
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    f"/telegram/webhook/{_VALID_TOKEN}",
                    json=_update(text="   "),
                )

        assert response.status_code == 200
        mock_bg.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: setup / delete webhook endpoints
# ---------------------------------------------------------------------------


class TestTelegramSetupEndpoints:
    @pytest.mark.asyncio
    async def test_setup_webhook_success(self):
        """POST /telegram/setup calls setWebhook and returns Telegram response."""
        telegram_response = {"ok": True, "description": "Webhook was set"}
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = telegram_response
        mock_http_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_http_response)

        with (
            patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.api.routers.telegram.httpx.AsyncClient", return_value=mock_client),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/telegram/setup",
                    json={"url": "https://abc.ngrok.io"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_setup_webhook_telegram_error(self):
        """POST /telegram/setup returns 502 when Telegram API fails."""
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_response)
        )

        with (
            patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.api.routers.telegram.httpx.AsyncClient", return_value=mock_client),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/telegram/setup",
                    json={"url": "https://abc.ngrok.io"},
                )

        assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_delete_webhook_success(self):
        """DELETE /telegram/setup calls deleteWebhook and returns Telegram response."""
        telegram_response = {"ok": True, "description": "Webhook was deleted"}
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = telegram_response
        mock_http_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_http_response)

        with (
            patch("app.api.routers.telegram.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.api.routers.telegram.httpx.AsyncClient", return_value=mock_client),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete("/telegram/setup")

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True


# ---------------------------------------------------------------------------
# Unit: background handler
# ---------------------------------------------------------------------------


class TestProcessTelegramMessageBackground:
    @pytest.mark.asyncio
    async def test_success_invokes_graph_and_sends_reply(self):
        """Happy path: graph is invoked and reply is sent."""
        mock_result = {"answer": "Pendaftaran dapat dilakukan di portal."}
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = mock_result

        with (
            patch("app.api.routers.chat._get_graph", return_value=mock_graph),
            patch("app.services.telegram_handler.send_telegram_reply") as mock_reply,
        ):
            mock_reply.return_value = None  # async no-op

            await process_telegram_message_background(
                chat_id=12345,
                text="Bagaimana cara mendaftar?",
                bot_token=_VALID_TOKEN,
            )

        mock_graph.ainvoke.assert_called_once()
        invoked_state = mock_graph.ainvoke.call_args.args[0]
        assert invoked_state["query"] == "Bagaimana cara mendaftar?"
        assert invoked_state["session_id"] == "tg_12345"

        mock_reply.assert_called_once_with(
            chat_id=12345,
            text="Pendaftaran dapat dilakukan di portal.",
            bot_token=_VALID_TOKEN,
        )

    @pytest.mark.asyncio
    async def test_graph_error_sends_fallback_reply(self):
        """When graph.ainvoke raises, a user-friendly fallback message is sent."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = RuntimeError("LLM timeout")

        with (
            patch("app.api.routers.chat._get_graph", return_value=mock_graph),
            patch("app.services.telegram_handler.send_telegram_reply") as mock_reply,
        ):
            mock_reply.return_value = None

            await process_telegram_message_background(
                chat_id=99999,
                text="Any question",
                bot_token=_VALID_TOKEN,
            )

        mock_reply.assert_called_once()
        sent_text: str = mock_reply.call_args.kwargs["text"]
        # Must be a non-empty fallback — not the crash traceback
        assert len(sent_text) > 10
        assert "error" in sent_text.lower() or "kesalahan" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_session_id_uses_tg_prefix(self):
        """Graph is invoked with tg_{chat_id} session_id, not a UUID."""
        chat_id = 777_888_999
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"answer": "ok"}

        with (
            patch("app.api.routers.chat._get_graph", return_value=mock_graph),
            patch("app.services.telegram_handler.send_telegram_reply") as mock_reply,
        ):
            mock_reply.return_value = None
            await process_telegram_message_background(
                chat_id=chat_id,
                text="Test",
                bot_token=_VALID_TOKEN,
            )

        state = mock_graph.ainvoke.call_args.args[0]
        assert state["session_id"] == f"tg_{chat_id}"
        # Ensure it is NOT a UUID format
        assert state["session_id"].startswith("tg_")
        assert "-" not in state["session_id"]

    @pytest.mark.asyncio
    async def test_empty_answer_uses_fallback_text(self):
        """If graph returns an empty answer, a default message is sent."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"answer": ""}

        with (
            patch("app.api.routers.chat._get_graph", return_value=mock_graph),
            patch("app.services.telegram_handler.send_telegram_reply") as mock_reply,
        ):
            mock_reply.return_value = None
            await process_telegram_message_background(
                chat_id=111,
                text="Test",
                bot_token=_VALID_TOKEN,
            )

        sent_text = mock_reply.call_args.kwargs["text"]
        assert len(sent_text) > 0

    @pytest.mark.asyncio
    async def test_send_telegram_reply_truncates_long_message(self):
        """send_telegram_reply must truncate text to 4096 chars (Telegram limit)."""
        from app.services.telegram_handler import send_telegram_reply

        long_text = "A" * 5000
        captured: list[dict] = []

        async def fake_post(url: str, json: dict) -> MagicMock:
            captured.append(json)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=fake_post)

        with patch("app.services.telegram_handler.httpx.AsyncClient", return_value=mock_client):
            await send_telegram_reply(chat_id=1, text=long_text, bot_token=_VALID_TOKEN)

        assert len(captured) == 1
        assert len(captured[0]["text"]) == 4096
