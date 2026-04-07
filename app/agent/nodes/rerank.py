"""Rerank node — uses Jina Reranker to sort docs by relevance."""

from __future__ import annotations

from app.agent.state import RerankInput, RerankUpdate
from app.config import get_settings
from app.services.reranker import rerank
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def rerank_docs(state: RerankInput) -> RerankUpdate:
    """Read retrieval docs and return reranked docs plus source refs."""
    query = state["query"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("No documents to rerank")
        return {"reranked_documents": [], "relevance_ok": False}

    reranked = await rerank(query=query, documents=documents)

    settings = get_settings()

    # Optionally drop chunks whose reranker score is negative
    if settings.filter_negative_scores:
        before = len(reranked)
        reranked = [d for d in reranked if d.get("relevance_score", 0.0) >= 0]
        dropped = before - len(reranked)
        if dropped:
            logger.info("Filtered %d negative-score chunks", dropped)

    # Check if the top document meets the relevance threshold
    top_score = reranked[0]["relevance_score"] if reranked else 0.0
    relevance_ok = top_score >= settings.relevance_threshold

    logger.info(
        "Reranked %d docs, top_score=%.4f, relevance_ok=%s",
        len(reranked),
        top_score,
        relevance_ok,
    )

    # Build rich source references matching SourceChunk schema
    sources = []
    seen_chunks: set[str] = set()
    for doc in reranked:
        chunk_key = f"{doc.get('doc_id', '')}:{doc.get('chunk_index', '')}"
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)
        sources.append(
            {
                "doc_id": doc.get("doc_id", "unknown"),
                "filename": doc.get("filename", "unknown"),
                "page": doc.get("page"),
                "score": doc.get("relevance_score"),
                "snippet": doc.get("text", "")[:150],
            }
        )

    return {
        "reranked_documents": reranked,
        "relevance_ok": relevance_ok,
        "sources": sources,
    }
