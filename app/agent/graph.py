"""Top-level LangGraph router for public and student assistants."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.classify import classify_query
from app.agent.nodes.memory import load_memory, store_memory
from app.agent.public_assistant import build_public_assistant
from app.agent.state import AgentState
from app.agent.student_assistant import build_student_assistant
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Routing functions ───────────────────────────────────


def route_after_classify(state: AgentState) -> str:
    """Delegate to the capability that owns the request."""
    route = state.get("route", "fallback")

    if route in ("fallback", "retrieval_only"):
        return "public_assistant"
    if route in ("student_only", "both", "nilai_semester"):
        return "student_assistant"
    return "public_assistant"


# ── Graph builder ───────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and return the compiled top-level LangGraph agent."""
    graph = StateGraph(AgentState)
    public_assistant = build_public_assistant()
    student_assistant = build_student_assistant()

    graph.add_node("load_memory", load_memory)
    graph.add_node("classify_query", classify_query)
    graph.add_node("public_assistant", public_assistant)
    graph.add_node("student_assistant", student_assistant)
    graph.add_node("store_memory", store_memory)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "classify_query")

    graph.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "public_assistant": "public_assistant",
            "student_assistant": "student_assistant",
        },
    )

    graph.add_edge("public_assistant", "store_memory")
    graph.add_edge("student_assistant", "store_memory")
    graph.add_edge("store_memory", END)

    compiled = graph.compile()
    logger.info("LangGraph agent compiled successfully")
    return compiled
