"""LangGraph StateGraph construction and compilation.

Graph flow (matches PROMPT.md mermaid diagram):

    START → load_memory → classify_query → [need_retrieval?]
      No  → generate_answer_fallback → store_memory → END
      Yes → retrieve_docs → rerank → [relevance_ok?]
        Yes → generate_answer → store_memory → END
        No  → rewrite_question → retrieve_docs  (loop, max 2 rewrites)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.classify import classify_query
from app.agent.nodes.generate import (
    generate_answer,
    generate_answer_fallback,
    rewrite_question,
)
from app.agent.nodes.memory import load_memory, store_memory
from app.agent.nodes.rerank import rerank_docs
from app.agent.nodes.retrieve import retrieve_docs
from app.agent.state import AgentState
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Routing functions ───────────────────────────────────


def route_after_classify(state: AgentState) -> str:
    """Route based on whether retrieval is needed."""
    if state.get("need_retrieval"):
        return "retrieve_docs"
    return "generate_answer_fallback"


def route_after_rerank(state: AgentState) -> str:
    """Route based on relevance check + rewrite count."""
    if state.get("relevance_ok"):
        return "generate_answer"

    rewrite_count = state.get("rewrite_count", 0)
    settings = get_settings()
    if rewrite_count >= settings.max_rewrite_count:
        logger.warning(
            "Max rewrites (%d) reached — generating with available context",
            rewrite_count,
        )
        return "generate_answer"

    return "rewrite_question"


# ── Graph builder ───────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph agent."""
    graph = StateGraph(AgentState)

    # ── Add nodes ───────────────────────────────────────
    graph.add_node("load_memory", load_memory)
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("rerank", rerank_docs)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("generate_answer_fallback", generate_answer_fallback)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("store_memory", store_memory)

    # ── Add edges ───────────────────────────────────────
    # START → load memory → classify
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "classify_query")

    # classify → conditional branch
    graph.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "retrieve_docs": "retrieve_docs",
            "generate_answer_fallback": "generate_answer_fallback",
        },
    )

    # retrieve → rerank → conditional branch
    graph.add_edge("retrieve_docs", "rerank")
    graph.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        },
    )

    # rewrite loops back to retrieve
    graph.add_edge("rewrite_question", "retrieve_docs")

    # both generate paths → store → END
    graph.add_edge("generate_answer", "store_memory")
    graph.add_edge("generate_answer_fallback", "store_memory")
    graph.add_edge("store_memory", END)

    compiled = graph.compile()
    logger.info("LangGraph agent compiled successfully")
    return compiled
