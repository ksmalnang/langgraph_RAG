"""Agent state definition for the LangGraph RAG agent."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph agent.

    Fields
    ------
    query : str
        The original (or rewritten) user query.
    session_id : str
        Chat session identifier for Redis history.
    chat_history : list[dict]
        Previous turns loaded from Redis.
    need_retrieval : bool
        Whether the query requires document retrieval.
    documents : list[dict]
        Raw documents returned from Qdrant search.
    reranked_documents : list[dict]
        Documents after Jina reranking.
    relevance_ok : bool
        Whether the top reranked doc exceeds the relevance threshold.
    rewrite_count : int
        How many times the query has been rewritten (max cap).
    answer : str
        The final generated answer.
    sources : list[dict]
        Source chunk references (doc_id, filename, score, snippet).
    """

    query: str
    session_id: str
    chat_history: list[dict[str, Any]]
    need_retrieval: bool
    documents: list[dict[str, Any]]
    reranked_documents: list[dict[str, Any]]
    relevance_ok: bool
    rewrite_count: int
    answer: str
    sources: list[dict[str, Any]]
