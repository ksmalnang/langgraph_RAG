"""Memory node — load/store chat history from Redis."""

from __future__ import annotations

from app.agent.state import AgentState
from app.services.memory import get_history, save_turn
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def load_memory(state: AgentState) -> dict:
    """Load chat history from Redis at the start of the flow."""
    session_id = state["session_id"]
    history = await get_history(session_id)
    logger.debug("Loaded %d history turns for session %s", len(history), session_id)
    return {"chat_history": history}


async def store_memory(state: AgentState) -> dict:
    """Persist the current turn (query + answer) to Redis."""
    session_id = state["session_id"]
    query = state["query"]
    answer = state.get("answer", "")

    await save_turn(session_id, user_message=query, assistant_message=answer)
    logger.debug("Stored turn for session %s", session_id)
    return {}
