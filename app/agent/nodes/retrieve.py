"""Retrieve documents node — embeds query and runs hybrid search."""

from __future__ import annotations

from app.agent.state import AgentState
from app.services.embeddings import embed_query
from app.services.vectorstore import hybrid_search
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def retrieve_docs(state: AgentState) -> dict:
    """Embed the query and run hybrid search (dense + BM25) via Qdrant."""
    query = state["query"]
    logger.info("Retrieving docs for: %s", query[:80])

    query_vector = await embed_query(query)
    documents = await hybrid_search(query_text=query, query_vector=query_vector)

    logger.info("Retrieved %d documents", len(documents))
    return {"documents": documents}
