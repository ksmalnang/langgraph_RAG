"""Tests for the LangGraph agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agent.state import AgentState


def test_agent_state_typing():
    """AgentState TypedDict has the expected keys."""
    state: AgentState = {
        "query": "test",
        "session_id": "s1",
        "chat_history": [],
        "need_retrieval": False,
        "documents": [],
        "reranked_documents": [],
        "relevance_ok": False,
        "rewrite_count": 0,
        "answer": "",
        "sources": [],
    }
    assert state["query"] == "test"
    assert state["rewrite_count"] == 0


def test_route_after_classify__public_routes():
    from app.agent.graph import route_after_classify

    assert route_after_classify({"route": "fallback"}) == "public_assistant"
    assert route_after_classify({"route": "retrieval_only"}) == "public_assistant"


def test_route_after_classify__student_routes():
    from app.agent.graph import route_after_classify

    assert route_after_classify({"route": "student_only"}) == "student_assistant"
    assert route_after_classify({"route": "both"}) == "student_assistant"
    assert route_after_classify({"route": "nilai_semester"}) == "student_assistant"


def test_public_assistant_routes():
    from app.agent.public_assistant import route_after_rerank, route_public_request

    assert route_public_request({"route": "fallback"}) == "generate_answer_fallback"
    assert route_public_request({"route": "retrieval_only"}) == "retrieve_docs"
    assert route_after_rerank({"relevance_ok": True}) == "generate_answer"


def test_student_assistant_routes():
    from app.agent.student_assistant import route_after_fetch

    assert route_after_fetch({"student_fetch_error": True}) == "generate_answer_fallback"
    assert route_after_fetch({"route": "nilai_semester"}) == "fetch_nilai_semester"
    assert route_after_fetch({"need_retrieval": True}) == "public_assistant"
    assert route_after_fetch({"need_retrieval": False}) == "generate_answer"


@pytest.mark.asyncio
async def test_classify_query_retrieval():
    """classify_query sets need_retrieval=True for admin questions."""
    from app.agent.nodes.classify import classify_query

    mock_response = AsyncMock()
    mock_response.content = '{"route": "retrieval_only", "reason": "Ask about process"}'

    with patch("app.agent.nodes.classify.get_llm") as mock_llm:
        mock_instance = AsyncMock()
        mock_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = mock_instance

        result = await classify_query({"query": "How to register?", "session_id": "s1"})

    assert set(result) == {"route", "need_retrieval"}
    assert result["need_retrieval"] is True


@pytest.mark.asyncio
async def test_classify_query_fallback():
    """classify_query sets need_retrieval=False for greetings."""
    from app.agent.nodes.classify import classify_query

    mock_response = AsyncMock()
    mock_response.content = '{"route": "fallback", "reason": "Just greeting"}'

    with patch("app.agent.nodes.classify.get_llm") as mock_llm:
        mock_instance = AsyncMock()
        mock_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = mock_instance

        result = await classify_query({"query": "Hello!", "session_id": "s1"})

    assert set(result) == {"route", "need_retrieval"}
    assert result["need_retrieval"] is False


@pytest.mark.asyncio
async def test_retrieve_docs():
    """retrieve_docs embeds and runs hybrid search."""
    from app.agent.nodes.retrieve import retrieve_docs

    with (
        patch(
            "app.agent.nodes.retrieve.embed_query", new_callable=AsyncMock
        ) as mock_embed,
        patch(
            "app.agent.nodes.retrieve.hybrid_search", new_callable=AsyncMock
        ) as mock_search,
    ):
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_search.return_value = [
            {"text": "doc1", "score": 0.9},
            {"text": "doc2", "score": 0.7},
        ]

        result = await retrieve_docs({"query": "test query", "session_id": "s1"})

    assert set(result) == {"documents"}
    assert len(result["documents"]) == 2
    mock_embed.assert_called_once_with("test query")
    mock_search.assert_called_once_with(
        query_text="test query", query_vector=[0.1, 0.2, 0.3]
    )


@pytest.mark.asyncio
async def test_store_memory():
    """store_memory saves the turn to Redis."""
    from app.agent.nodes.memory import store_memory

    with patch("app.agent.nodes.memory.save_turn", new_callable=AsyncMock) as mock_save:
        await store_memory({"session_id": "s1", "query": "Hi", "answer": "Hello!"})

    mock_save.assert_called_once_with(
        "s1", user_message="Hi", assistant_message="Hello!"
    )


@pytest.mark.asyncio
async def test_rerank_docs_returns_owned_fields_only():
    from app.agent.nodes.rerank import rerank_docs

    with (
        patch("app.agent.nodes.rerank.rerank", new_callable=AsyncMock) as mock_rerank,
        patch("app.agent.nodes.rerank.get_settings") as mock_settings,
    ):
        mock_rerank.return_value = [
            {
                "doc_id": "doc-1",
                "chunk_index": 0,
                "filename": "guide.pdf",
                "page": 3,
                "text": "helpful context",
                "relevance_score": 0.91,
            }
        ]
        mock_settings.return_value.filter_negative_scores = False
        mock_settings.return_value.relevance_threshold = 0.5

        result = await rerank_docs(
            {
                "query": "How do I register?",
                "documents": [{"text": "helpful context"}],
            }
        )

    assert set(result) == {"reranked_documents", "relevance_ok", "sources"}
    assert result["relevance_ok"] is True
    assert result["sources"][0]["doc_id"] == "doc-1"


@pytest.mark.asyncio
async def test_graph_compiles():
    """The agent graph compiles without errors."""
    with patch("app.agent.public_assistant.get_settings") as mock_settings:
        mock_settings.return_value.max_rewrite_count = 2
        from app.agent.graph import build_graph

        graph = build_graph()
        assert graph is not None


@pytest.mark.asyncio
async def test_load_memory():
    """load_memory fetches the turn history from Redis."""
    from app.agent.nodes.memory import load_memory

    with patch(
        "app.agent.nodes.memory.get_history", new_callable=AsyncMock
    ) as mock_get:
        mock_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        mock_get.return_value = mock_history

        result = await load_memory({"session_id": "s1", "query": "What's up?"})

    mock_get.assert_called_once_with("s1")
    assert result["chat_history"] == mock_history


@pytest.mark.asyncio
async def test_generate_answer_uses_history():
    """generate_answer_fallback formats chat history into the LLM prompt."""
    from app.agent.nodes.generate import generate_answer_fallback

    mock_response = AsyncMock()
    mock_response.content = "Multi-turn answer"

    with patch("app.agent.nodes.generate.get_llm") as mock_llm:
        mock_instance = AsyncMock()
        mock_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = mock_instance

        state: AgentState = {
            "query": "And then?",
            "session_id": "s1",
            "chat_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }

        result = await generate_answer_fallback(state)

    assert result["answer"] == "Multi-turn answer"

    # Check that LLM was called
    assert mock_instance.ainvoke.call_count == 1
    call_args = mock_instance.ainvoke.call_args[0][0]  # The messages list

    # Verify that the system message contains the formatted history
    system_message = call_args[0]
    assert system_message.type == "system"
    assert "User: Hello" in system_message.content
    assert "Assistant: Hi there!" in system_message.content
