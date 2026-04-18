# app/agent/state.py

from __future__ import annotations

from typing import Any, Literal, NotRequired, Required

from typing_extensions import TypedDict

RouteName = Literal[
    "fallback",
    "retrieval_only",
    "student_only",
    "both",
    "nilai_semester",
]


class ChatTurn(TypedDict, total=False):
    """Single chat turn stored in conversation history."""

    role: Required[str]
    content: Required[str]
    timestamp: NotRequired[str | None]
    message_id: NotRequired[str | None]


class SourceReference(TypedDict, total=False):
    """Minimal source payload returned by the rerank step and API."""

    doc_id: Required[str]
    filename: Required[str]
    page: NotRequired[int | None]
    score: NotRequired[float | None]
    snippet: NotRequired[str]


class SharedAgentState(TypedDict, total=False):
    """State shared across all assistant capabilities.

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

    """

    query: Required[str]
    session_id: Required[str]
    answer: NotRequired[str]
    sources: NotRequired[list[SourceReference]]

    chat_history: NotRequired[list[ChatTurn]]

    route: NotRequired[RouteName]
    need_retrieval: NotRequired[bool]
    message_id: NotRequired[str]


class PublicAssistantState(TypedDict, total=False):
    """State owned by the public/admin retrieval assistant."""

    documents: list[dict[str, Any]]
    reranked_documents: list[dict[str, Any]]
    relevance_ok: bool
    rewrite_count: int


class StudentAssistantState(TypedDict, total=False):
    """State owned by the authenticated student/SIAKAD assistant."""

    student_data: dict[str, Any] | None
    student_fetch_error: bool
    nilai_semester_detail: dict | None


class AgentState(
    SharedAgentState,
    PublicAssistantState,
    StudentAssistantState,
    total=False,
):
    """Combined LangGraph state used by the top-level router graph."""


class ClassificationInput(TypedDict, total=False):
    """Inputs required by the classify node."""

    query: Required[str]
    chat_history: NotRequired[list[ChatTurn]]


class ClassificationUpdate(TypedDict):
    """Fields written by the classify node."""

    route: RouteName
    need_retrieval: bool


class RetrieveInput(TypedDict):
    """Inputs required by the retrieval node."""

    query: str


class RetrieveUpdate(TypedDict):
    """Fields written by the retrieval node."""

    documents: list[dict[str, Any]]


class RerankInput(TypedDict, total=False):
    """Inputs required by the rerank node."""

    query: Required[str]
    documents: NotRequired[list[dict[str, Any]]]


class RerankUpdate(TypedDict):
    """Fields written by the rerank node."""

    reranked_documents: list[dict[str, Any]]
    relevance_ok: bool
    sources: list[SourceReference]


class RewriteInput(TypedDict, total=False):
    """Inputs required by the query rewrite node."""

    query: Required[str]
    rewrite_count: NotRequired[int]


class RewriteUpdate(TypedDict):
    """Fields written by the query rewrite node."""

    query: str
    rewrite_count: int


class GenerateAnswerInput(TypedDict, total=False):
    """Inputs consumed when assembling the final prompt."""

    query: Required[str]
    chat_history: NotRequired[list[ChatTurn]]
    reranked_documents: NotRequired[list[dict[str, Any]]]
    student_data: NotRequired[dict[str, Any] | None]
    nilai_semester_detail: NotRequired[dict[str, Any] | None]


class GenerateAnswerUpdate(TypedDict):
    """Fields written by answer-generation nodes."""

    answer: str


class GenerateFallbackUpdate(GenerateAnswerUpdate, total=False):
    """Fallback generation also owns clearing source references."""

    sources: list[SourceReference]


class FetchStudentInput(TypedDict):
    """Inputs required by the SIAKAD fetch node."""

    session_id: str


class FetchStudentUpdate(TypedDict):
    """Fields written by the SIAKAD fetch node."""

    student_data: dict[str, Any] | None
    student_fetch_error: bool
