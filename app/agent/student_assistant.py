"""Authenticated student assistant orchestration."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.fetch import fetch_student_data
from app.agent.nodes.fetch_nilai_semester import fetch_nilai_semester
from app.agent.nodes.generate import generate_answer, generate_answer_fallback
from app.agent.public_assistant import build_public_assistant
from app.agent.state import AgentState


def route_after_fetch(state: AgentState) -> str:
    """Route after attempting to fetch student data."""
    if state.get("student_fetch_error"):
        return "generate_answer_fallback"

    if state.get("route") == "nilai_semester":
        return "fetch_nilai_semester"

    if state.get("need_retrieval"):
        return "public_assistant"

    return "generate_answer"


def build_student_assistant():
    """Build the student/SIAKAD assistant subgraph."""
    graph = StateGraph(AgentState)

    graph.add_node("fetch_student_data", fetch_student_data)
    graph.add_node("fetch_nilai_semester", fetch_nilai_semester)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("generate_answer_fallback", generate_answer_fallback)
    graph.add_node("public_assistant", build_public_assistant())

    graph.add_edge(START, "fetch_student_data")
    graph.add_conditional_edges(
        "fetch_student_data",
        route_after_fetch,
        {
            "generate_answer_fallback": "generate_answer_fallback",
            "public_assistant": "public_assistant",
            "generate_answer": "generate_answer",
            "fetch_nilai_semester": "fetch_nilai_semester",
        },
    )
    graph.add_edge("fetch_nilai_semester", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("generate_answer_fallback", END)
    graph.add_edge("public_assistant", END)

    return graph.compile()
