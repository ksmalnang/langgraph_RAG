"""Qdrant async vector store wrapper with hybrid search (dense + BM25)."""

from __future__ import annotations

from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from app.config import get_settings
from app.services.resilience import elapsed_ms, now_ms, retry_async
from app.utils.exceptions import VectorStoreError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level client — initialised lazily on first access.
_client: AsyncQdrantClient | None = None

# Lazy singleton for the BM25 sparse encoder (fastembed).
_bm25_encoder: SparseTextEmbedding | None = None


def _get_bm25_encoder() -> SparseTextEmbedding:
    """Return (or create) the singleton BM25 sparse text encoder."""
    global _bm25_encoder
    if _bm25_encoder is None:
        _bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("Loaded BM25 sparse encoder (Qdrant/bm25)")
    return _bm25_encoder


def _encode_sparse(texts: list[str]) -> list[models.SparseVector]:
    """Encode texts into BM25 sparse vectors using fastembed."""
    encoder = _get_bm25_encoder()
    results: list[models.SparseVector] = []
    for embedding in encoder.embed(texts):
        results.append(
            models.SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
        )
    return results


def _encode_sparse_query(text: str) -> models.SparseVector:
    """Encode a single query into a BM25 sparse vector."""
    encoder = _get_bm25_encoder()
    embedding = next(encoder.query_embed(text))
    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


async def get_qdrant_client() -> AsyncQdrantClient:
    """Return (or create) the singleton async Qdrant client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout_seconds,
        )
    return _client


async def ensure_collection(vector_size: int) -> None:
    """Create the collection with named dense + BM25 sparse vectors.

    Uses named vectors so that hybrid search can target each arm
    independently via ``prefetch``.
    """
    settings = get_settings()
    client = await get_qdrant_client()
    start_ms = now_ms()
    try:
        collections = await retry_async(
            operation="get_collections",
            dependency="qdrant",
            fn=client.get_collections,
            retry_on=(ResponseHandlingException, TimeoutError),
        )
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=qdrant operation=get_collections mode=unexpected latency_ms=%.1f",
            elapsed_ms(start_ms),
            exc_info=True,
        )
        raise VectorStoreError("Failed to read Qdrant collections") from exc
    names = [c.name for c in collections.collections]

    if settings.collection_name not in names:
        start_ms = now_ms()
        try:
            await client.create_collection(
                collection_name=settings.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
        except Exception as exc:
            logger.error(
                "Dependency failure dependency=qdrant operation=create_collection mode=unexpected latency_ms=%.1f",
                elapsed_ms(start_ms),
                exc_info=True,
            )
            raise VectorStoreError("Failed to create Qdrant collection") from exc
        logger.info(
            "Created Qdrant collection '%s' (dense dim=%d + BM25 sparse)",
            settings.collection_name,
            vector_size,
        )
    else:
        logger.debug("Collection '%s' already exists", settings.collection_name)


async def upsert_points(
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
) -> None:
    """Upsert points with named dense vectors + BM25 sparse vectors."""
    settings = get_settings()
    client = await get_qdrant_client()

    """Generate BM25 sparse vectors client-side via fastembed.

    Prefer the heading-enriched text so sparse retrieval also benefits
    from section context; fall back to raw text for backward compat.
    """

    texts = [pay.get("enriched_text", pay["text"]) for pay in payloads]
    sparse_vectors = _encode_sparse(texts)

    points = [
        models.PointStruct(
            id=uid,
            vector={
                "dense": vec,
                "sparse": sparse,
            },
            payload=pay,
        )
        for uid, vec, sparse, pay in zip(
            ids, vectors, sparse_vectors, payloads, strict=False
        )
    ]
    start_ms = now_ms()
    try:
        await client.upsert(
            collection_name=settings.collection_name,
            points=points,
        )
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=qdrant operation=upsert mode=unexpected points=%d latency_ms=%.1f",
            len(points),
            elapsed_ms(start_ms),
            exc_info=True,
        )
        raise VectorStoreError("Failed to upsert vectors to Qdrant") from exc
    logger.info("Upserted %d points into '%s'", len(points), settings.collection_name)


async def delete_points_by_doc_id(doc_id: str) -> None:
    """Delete all points belonging to a single document identity."""
    settings = get_settings()
    client = await get_qdrant_client()

    selector = models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
    )

    start_ms = now_ms()
    try:
        await client.delete(
            collection_name=settings.collection_name,
            points_selector=selector,
            wait=True,
        )
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=qdrant operation=delete_doc_points mode=unexpected doc_id=%s latency_ms=%.1f",
            doc_id,
            elapsed_ms(start_ms),
            exc_info=True,
        )
        raise VectorStoreError("Failed to delete existing document points") from exc

    logger.info("Deleted previous points for doc_id=%s", doc_id)


async def hybrid_search(
    query_text: str,
    query_vector: list[float],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search: BM25 sparse + dense prefetch, fused with RRF.

    Both arms independently retrieve ``prefetch_limit`` candidates,
    then Qdrant fuses them with Reciprocal Rank Fusion.
    """
    settings = get_settings()
    client = await get_qdrant_client()
    top_k = top_k or settings.retrieval_top_k
    prefetch_limit = settings.hybrid_prefetch_limit

    # Generate BM25 sparse query vector client-side via fastembed
    sparse_query = _encode_sparse_query(query_text)

    start_ms = now_ms()

    async def _query():
        return await client.query_points(
            collection_name=settings.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

    try:
        results = await retry_async(
            operation="query_points",
            dependency="qdrant",
            fn=_query,
            retry_on=(ResponseHandlingException, TimeoutError),
        )
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=qdrant operation=query_points mode=unexpected latency_ms=%.1f",
            elapsed_ms(start_ms),
            exc_info=True,
        )
        raise VectorStoreError("Failed to query Qdrant") from exc

    docs: list[dict[str, Any]] = []
    for point in results.points:
        doc = dict(point.payload) if point.payload else {}
        doc["score"] = point.score
        doc["id"] = str(point.id)
        docs.append(doc)

    logger.debug("Hybrid search returned %d results", len(docs))
    return docs


async def close_client() -> None:
    """Close the Qdrant client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
