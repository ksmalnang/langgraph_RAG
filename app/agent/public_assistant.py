"""Public/admin assistant orchestration."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.generate import (
    generate_answer,
    generate_answer_fallback,
    rewrite_question,
)
from app.agent.nodes.rerank import rerank_docs
from app.agent.nodes.retrieve import retrieve_docs
from app.agent.state import AgentState
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def route_public_request(state: AgentState) -> str:
    """Choose the public/admin path after classification."""
    if state.get("route") == "fallback":
        return "generate_answer_fallback"
    return "retrieve_docs"


def route_after_rerank(state: AgentState) -> str:
    """Route based on relevance check + rewrite count."""
    if state.get("relevance_ok"):
        return "generate_answer"

    rewrite_count = state.get("rewrite_count", 0)
    settings = get_settings()
    if rewrite_count >= settings.max_rewrite_count:
        logger.warning(
            "Max rewrites (%d) reached; generating fallback answer",
            rewrite_count,
        )
        return "generate_answer_fallback"

    return "rewrite_question"


def build_public_assistant():
    """Build the public/admin assistant subgraph."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("rerank", rerank_docs)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("generate_answer_fallback", generate_answer_fallback)
    graph.add_node("rewrite_question", rewrite_question)

    graph.add_conditional_edges(
        START,
        route_public_request,
        {
            "generate_answer_fallback": "generate_answer_fallback",
            "retrieve_docs": "retrieve_docs",
        },
    )

    graph.add_edge("retrieve_docs", "rerank")
    graph.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {
            "generate_answer": "generate_answer",
            "generate_answer_fallback": "generate_answer_fallback",
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_docs")
    graph.add_edge("generate_answer", END)
    graph.add_edge("generate_answer_fallback", END)

    return graph.compile()
