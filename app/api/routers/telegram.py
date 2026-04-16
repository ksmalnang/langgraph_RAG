"""Telegram Bot Webhook endpoints.

Routes
------
POST /telegram/webhook/{bot_token}
    Primary receiver for Telegram update POSTs.  Verifies the token,
    deduplicates retries via Redis, acknowledges immediately (200 OK),
    and queues LLM processing as a FastAPI BackgroundTask.

POST /telegram/setup
    Programmatically register a public URL as the Telegram webhook.

DELETE /telegram/setup
    Unregister the current webhook (deleteWebhook).

Security
--------
The ``bot_token`` path parameter acts as a shared secret: only Telegram
(which knows the correct token) can trigger meaningful processing.

Session IDs
-----------
Telegram chat IDs are 64-bit integers.  We **never** use
``_resolve_session_id`` from ``chat.py`` (which requires UUID v4 strings).
Instead we coerce to ``tg_{chat_id}`` via
``app.services.telegram_handler.build_telegram_session_id``.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from app.api.models import TelegramUpdate, WebhookSetupRequest
from app.config import get_settings
from app.services.memory import get_redis
from app.services.telegram_handler import process_telegram_message_background
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])

# Redis key prefix for processed update IDs (deduplication).
_DEDUP_KEY_PREFIX = "tg:dedup:update_id:"
# TTL for dedup keys: Telegram retries within ~1 minute; 5 minutes is safe.
_DEDUP_TTL_SECONDS = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _is_duplicate_update(update_id: int) -> bool:
    """Return True if this update_id was already processed; mark it if not.

    Uses a Redis SET NX (set-if-not-exists) to atomically check *and* mark
    the update in a single round-trip.

    Args:
        update_id: The Telegram ``update_id`` from the incoming payload.

    Returns:
        ``True`` if this is a duplicate (key already existed), ``False``
        if it is new (key was set successfully).
    """
    try:
        r = await get_redis()
        key = f"{_DEDUP_KEY_PREFIX}{update_id}"
        # SET NX returns True if the key was newly set, False if it existed.
        was_set: bool = await r.set(key, "1", ex=_DEDUP_TTL_SECONDS, nx=True)
        return not was_set  # duplicate if it was NOT newly set
    except Exception:
        # If Redis is unavailable we err on the side of processing the message
        # rather than silently dropping it.
        logger.exception("Redis dedup check failed for update_id=%d; processing anyway", update_id)
        return False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/webhook/{bot_token}")
async def telegram_webhook(
    bot_token: str,
    update: TelegramUpdate,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Receive and dispatch a Telegram update.

    1. Verify the token matches the configured ``TELEGRAM_BOT_TOKEN``.
    2. Deduplicate retries using the ``update_id`` stored in Redis.
    3. Drop non-text messages silently (stickers, photos, etc.).
    4. Enqueue AI processing as a BackgroundTask and return 200 immediately.
    """
    settings = get_settings()

    # ── Token verification ────────────────────────────────────────────────
    if bot_token != settings.telegram_bot_token:
        logger.warning("Telegram webhook received invalid token — ignoring update")
        # Return 200 to avoid Telegram retrying indefinitely.
        return JSONResponse(content={"ok": True})

    # ── Deduplication ─────────────────────────────────────────────────────
    if await _is_duplicate_update(update.update_id):
        logger.info("Duplicate Telegram update_id=%d — dropping", update.update_id)
        return JSONResponse(content={"ok": True})

    # ── Non-text filter ───────────────────────────────────────────────────
    message = update.message
    if message is None or not message.text:
        # Silently acknowledge stickers, photos, voice messages, etc.
        logger.debug(
            "Non-text Telegram update update_id=%d — ignoring",
            update.update_id,
        )
        return JSONResponse(content={"ok": True})

    chat_id = message.chat.id
    text = message.text.strip()

    if not text:
        return JSONResponse(content={"ok": True})

    logger.info(
        "Telegram update queued update_id=%d chat_id=%d text_length=%d",
        update.update_id,
        chat_id,
        len(text),
    )

    # ── Queue background processing ───────────────────────────────────────
    background_tasks.add_task(
        process_telegram_message_background,
        chat_id=chat_id,
        text=text,
        bot_token=bot_token,
    )

    # Telegram requires a fast 200 OK; the AI reply is sent asynchronously.
    return JSONResponse(content={"ok": True})


@router.post("/setup")
async def setup_webhook(payload: WebhookSetupRequest) -> JSONResponse:
    """Register a public HTTPS URL as the Telegram Bot webhook.

    Typically used with a tunnelling tool such as ngrok during local
    development to expose the FastAPI server to Telegram's servers.

    Args:
        payload: Contains the public ``url`` to bind.

    Returns:
        The raw JSON response from the Telegram Bot API.
    """
    settings = get_settings()
    token = settings.telegram_bot_token
    webhook_url = f"{payload.url}/telegram/webhook/{token}"

    api_url = f"https://api.telegram.org/bot{token}/setWebhook"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(api_url, json={"url": webhook_url})
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "setWebhook failed: status=%d body=%s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        return JSONResponse(
            status_code=502,
            content={"ok": False, "description": "Telegram API error during setWebhook"},
        )
    except Exception:
        logger.exception("Unexpected error during setWebhook")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "description": "Internal error during webhook setup"},
        )

    logger.info("Webhook registered: url=%s result=%s", webhook_url, data.get("description"))
    return JSONResponse(content=data)


@router.delete("/setup")
async def delete_webhook() -> JSONResponse:
    """Unregister the current Telegram Bot webhook (deleteWebhook).

    Useful when switching from webhook mode back to polling, or when
    tearing down a development environment.

    Returns:
        The raw JSON response from the Telegram Bot API.
    """
    settings = get_settings()
    token = settings.telegram_bot_token
    api_url = f"https://api.telegram.org/bot{token}/deleteWebhook"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(api_url)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "deleteWebhook failed: status=%d body=%s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        return JSONResponse(
            status_code=502,
            content={"ok": False, "description": "Telegram API error during deleteWebhook"},
        )
    except Exception:
        logger.exception("Unexpected error during deleteWebhook")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "description": "Internal error during webhook teardown"},
        )

    logger.info("Webhook deleted: result=%s", data.get("description"))
    return JSONResponse(content=data)
