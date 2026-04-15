"""Background processing and reply helper for the Telegram webhook integration.

This module intentionally avoids using ``_resolve_session_id`` from
``chat.py`` because that function assumes UUIDv4 strings.  Telegram chat
identifiers are 64-bit integers which we represent as the prefixed string
``tg_{chat_id}`` — always with an underscore — to maintain consistent
thread histories inside the Redis checkpointer used by LangGraph.
"""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"
_SEND_MESSAGE_PATH = "/sendMessage"

# ---------------------------------------------------------------------------
# Session ID helpers
# ---------------------------------------------------------------------------


def build_telegram_session_id(chat_id: int) -> str:
    """Return a stable, Redis-safe session key for a Telegram chat.

    Args:
        chat_id: The 64-bit Telegram chat identifier.

    Returns:
        A string in the form ``tg_{chat_id}`` (e.g. ``tg_123456789``).
    """
    return f"tg_{chat_id}"


# ---------------------------------------------------------------------------
# Telegram reply helper
# ---------------------------------------------------------------------------


async def send_telegram_reply(
    chat_id: int,
    text: str,
    bot_token: str,
) -> None:
    """Send a text reply to a Telegram chat.

    The message is sent as plain text (no ``parse_mode``) to avoid strict
    MarkdownV2 / HTML escaping issues with arbitrary LangGraph output.

    Args:
        chat_id:   Telegram chat ID to reply to.
        text:      Message text (will be truncated to 4096 chars if needed).
        bot_token: Telegram Bot API token.
    """
    url = f"{_TELEGRAM_API_BASE.format(token=bot_token)}{_SEND_MESSAGE_PATH}"
    payload = {
        "chat_id": chat_id,
        # Telegram hard-cap is 4096 chars per message.
        "text": text[:4096],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Telegram sendMessage HTTP error: status=%d body=%s",
            exc.response.status_code,
            exc.response.text[:200],
        )
    except Exception:
        logger.exception("Unexpected error while sending Telegram reply to chat_id=%d", chat_id)


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def process_telegram_message_background(
    chat_id: int,
    text: str,
    bot_token: str,
) -> None:
    """Invoke the LangGraph AI pipeline and send the reply back to Telegram.

    This function is designed to run inside a FastAPI ``BackgroundTask``.
    Because FastAPI swallows ``BackgroundTask`` exceptions silently, **all**
    errors are caught here and surfaced as a fallback Telegram reply so the
    user is never left waiting with no response.

    Args:
        chat_id:   Telegram chat ID that sent the message.
        text:      Cleaned user message text.
        bot_token: Telegram Bot API token (used for the reply call).
    """
    logger.info(
        "Processing telegram message chat_id=%d text_length=%d",
        chat_id,
        len(text),
    )

    session_id = build_telegram_session_id(chat_id)

    # Build initial graph state — mirrors _build_initial_state() in chat.py but
    # uses the Telegram session_id format.  No SIAKAD auth is involved; guests
    # (unauthenticated interactions) are fully supported by the graph.
    initial_state: dict = {
        "query": text,
        "session_id": session_id,
        "chat_history": [],
        "need_retrieval": False,
        "documents": [],
        "reranked_documents": [],
        "relevance_ok": False,
        "rewrite_count": 0,
        "answer": "",
        "sources": [],
    }

    try:
        from app.api.routers.chat import _get_graph  # noqa: PLC0415

        graph = _get_graph()
        result = await graph.ainvoke(initial_state)
        answer: str = result.get("answer") or "Maaf, saya tidak dapat menghasilkan jawaban."
    except Exception:
        logger.exception(
            "LangGraph invocation failed for telegram chat_id=%d session_id=%s",
            chat_id,
            session_id,
        )
        answer = (
            "⚠️ Terjadi kesalahan internal saat memproses pesan Anda. "
            "Silakan coba lagi beberapa saat."
        )

    await send_telegram_reply(chat_id=chat_id, text=answer, bot_token=bot_token)
