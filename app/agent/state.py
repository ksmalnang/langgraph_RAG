# app/agent/state.py

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State flowing through the LangGraph agent.

    Fields
    ------
    query : str
        The original user query.
    session_id : str
        Chat session identifier (used for Redis history).
    answer : str
        The final generated answer.
    sources : list[dict[str, Any]]
        Source chunk references (doc_id, filename, score, snippet).

    chat_history : list[dict[str, Any]]
        Previous turns loaded from Redis (or other memory).

    route : Literal["fallback", "retrieval_only", "student_only", "both"]
        Determined by `classify_query`, read by `route_after_classify`.

    student_data : dict[str, Any] | None
        Student data fetched by `fetch_student_data` (if any).
    student_fetch_error : bool
        Whether an error occurred while fetching student data.

    need_retrieval : bool
        Preserved from `route`; indicates if document retrieval is required.
    documents : list[dict[str, Any]]
        Raw documents returned from Qdrant search.
    reranked_documents : list[dict[str, Any]]
        Documents after Jina reranking.
    relevance_ok : bool
        Whether the top reranked document exceeds the relevance threshold.
    rewrite_count : int
        How many times the query has been rewritten (max cap).
    """

    # ============================================================
    # Main I/O
    # ============================================================
    query: str
    session_id: str
    answer: str
    sources: list[dict[str, Any]]

    # ============================================================
    # Memory
    # ============================================================
    chat_history: list[dict[str, Any]]

    # ============================================================
    # Routing
    # Set by classify_query, read by route_after_classify
    # ============================================================
    route: Literal[
        "fallback", "retrieval_only", "student_only", "both", "nilai_semester"
    ]

    # ============================================================
    # Student Data
    # Set by fetch_student_data
    # ============================================================
    student_data: dict[str, Any] | None
    student_fetch_error: bool
    nilai_semester_detail: dict | None

    # ============================================================
    # Retrieval
    # ============================================================
    need_retrieval: bool  # preserved from route, read post-fetch
    documents: list[dict[str, Any]]
    reranked_documents: list[dict[str, Any]]
    relevance_ok: bool
    rewrite_count: int
