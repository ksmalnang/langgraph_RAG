"""Generate answer nodes — with and without document context."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import (
    FALLBACK_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
)
from app.agent.state import AgentState
from app.services.llm import get_llm, get_llm_cheap
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _format_history(chat_history: list[dict]) -> str:
    """Format chat history into a readable string."""
    if not chat_history:
        return "(no previous conversation)"
    lines = []
    for turn in chat_history[-10:]:  # Keep last 10 turns for context window
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_context(documents: list[dict]) -> str:
    """Format reranked documents into a context block."""
    if not documents:
        return "(no documents)"
    parts = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        score = doc.get("relevance_score", 0.0)
        headings = doc.get("headings", [])
        heading_str = " > ".join(headings) if headings else "N/A"
        parts.append(f"[{i}] (relevance: {score:.2f}, section: {heading_str})\n{text}")
    return "\n\n".join(parts)


async def generate_answer(state: AgentState) -> dict:
    """Generate an answer with document context (RAG path)."""
    query = state["query"]
    history = state.get("chat_history", [])
    documents = state.get("reranked_documents", [])

    context_str = _format_context(documents)
    history_str = _format_history(history)

    system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str, history=history_str)

    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    logger.info("Generating RAG answer for: %s", query[:80])
    response = await llm.ainvoke(messages)
    answer = response.content.strip()

    logger.debug("Generated answer length: %d chars", len(answer))
    return {"answer": answer}


async def generate_answer_fallback(state: AgentState) -> dict:
    """Generate an answer without document context (fallback path)."""
    query = state["query"]
    history = state.get("chat_history", [])

    history_str = _format_history(history)
    system_prompt = FALLBACK_SYSTEM_PROMPT.format(history=history_str)

    llm = get_llm(temperature=0.5)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    logger.info("Generating fallback answer for: %s", query[:80])
    response = await llm.ainvoke(messages)
    answer = response.content.strip()

    return {"answer": answer, "sources": []}


async def rewrite_question(state: AgentState) -> dict:
    """Rewrite the query for better retrieval results."""
    query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)

    llm = get_llm_cheap(temperature=0.1)
    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    logger.info("Rewriting query (attempt %d): %s", rewrite_count + 1, query[:80])
    response = await llm.ainvoke(messages)
    rewritten = response.content.strip()

    logger.info("Rewritten query: %s", rewritten[:80])
    return {"query": rewritten, "rewrite_count": rewrite_count + 1}
