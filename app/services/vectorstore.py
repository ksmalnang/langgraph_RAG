"""Qdrant async vector store wrapper with hybrid search (dense + BM25)."""

from __future__ import annotations

from collections import defaultdict
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
        logger.exception(
            "Dependency failure dependency=qdrant operation=get_collections mode=unexpected latency_ms=%.1f",
            elapsed_ms(start_ms),
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
            logger.exception(
                "Dependency failure dependency=qdrant operation=create_collection mode=unexpected latency_ms=%.1f",
                elapsed_ms(start_ms),
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
                "bm25": sparse,  # Match collection config sparse vector name
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
        logger.exception(
            "Dependency failure dependency=qdrant operation=upsert mode=unexpected points=%d latency_ms=%.1f",
            len(points),
            elapsed_ms(start_ms),
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
        logger.exception(
            "Dependency failure dependency=qdrant operation=delete_doc_points mode=unexpected doc_id=%s latency_ms=%.1f",
            doc_id,
            elapsed_ms(start_ms),
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
                    using="bm25",
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
        logger.exception(
            "Dependency failure dependency=qdrant operation=query_points mode=unexpected latency_ms=%.1f",
            elapsed_ms(start_ms),
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


async def list_files() -> list[dict[str, Any]]:
    """List all unique ingested files with their chunk counts.

    Scrolls all points in the collection, deduplicates by ``doc_id``,
    and returns one entry per file with metadata extracted from chunk payloads.

    Returns a list of dicts sorted by filename ascending, each containing:
    - doc_id
    - filename
    - doc_category
    - academic_year
    - total_chunks
    """
    settings = get_settings()
    client = await get_qdrant_client()

    # Scroll all points — Option A (dev/small scale)
    # Using a large enough limit to fetch everything in one go
    start_ms = now_ms()
    try:
        points, _next_offset = await client.scroll(
            collection_name=settings.collection_name,
            limit=100_000,
            with_payload=True,
        )
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=scroll mode=unexpected latency_ms=%.1f",
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to list files from Qdrant") from exc

    # Deduplicate by doc_id and count chunks per doc_id
    doc_map: dict[str, dict[str, Any]] = {}
    chunk_counts: dict[str, int] = defaultdict(int)

    for point in points:
        payload = point.payload or {}
        doc_id = payload.get("doc_id")
        if doc_id is None:
            continue

        chunk_counts[doc_id] += 1

        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "doc_id": doc_id,
                "filename": payload.get("filename", ""),
                "doc_category": payload.get("doc_category"),
                "academic_year": payload.get("academic_year"),
            }

    # Build the result with total_chunks
    files = []
    for doc_id, meta in doc_map.items():
        files.append(
            {
                **meta,
                "total_chunks": chunk_counts[doc_id],
            }
        )

    # Sort by filename ascending for stable ordering
    files.sort(key=lambda f: f["filename"])

    logger.info("Listed %d unique files from Qdrant", len(files))
    return files


async def delete_file(doc_id: str) -> dict[str, Any]:
    """Delete all chunks belonging to a single ingested file.

    First counts the chunks and extracts metadata, then performs the deletion.
    Returns a dict with doc_id, filename, deleted_chunks count.

    Raises:
        VectorStoreError: If the doc_id is not found (to be mapped to 404).
    """
    settings = get_settings()
    client = await get_qdrant_client()

    # Step 1: Count chunks and get metadata before deletion
    start_ms = now_ms()
    try:
        points, _next_offset = await client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=100_000,
            with_payload=True,
        )
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=scroll_delete mode=unexpected doc_id=%s latency_ms=%.1f",
            doc_id,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to query file before deletion") from exc

    chunk_count = len(points)
    if chunk_count == 0:
        raise VectorStoreError(f"File with doc_id='{doc_id}' not found")

    # Extract filename from the first point's payload
    filename = points[0].payload.get("filename", "") if points[0].payload else ""

    # Step 2: Delete all points matching doc_id
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
        logger.exception(
            "Dependency failure dependency=qdrant operation=delete_file mode=unexpected doc_id=%s latency_ms=%.1f",
            doc_id,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to delete file from Qdrant") from exc

    logger.info(
        "Deleted file doc_id=%s (%s) with %d chunks", doc_id, filename, chunk_count
    )
    return {
        "doc_id": doc_id,
        "filename": filename,
        "deleted_chunks": chunk_count,
    }


async def scroll_chunks_by_doc_id(doc_id: str) -> list[dict[str, Any]]:
    """Scroll all chunks for a given ``doc_id``, handling full pagination.

    Uses Qdrant scroll with offset-based pagination until ``next_offset`` is
    ``None``. Returns all points sorted by ``chunk_index``.

    Raises:
        VectorStoreError: If the query fails.
    """
    settings = get_settings()
    client = await get_qdrant_client()

    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
    )

    all_points: list[models.Record] = []
    offset = None

    start_ms = now_ms()
    try:
        while True:
            results, next_offset = await client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=scroll_chunks mode=unexpected doc_id=%s latency_ms=%.1f",
            doc_id,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to retrieve chunks from Qdrant") from exc

    # Sort by chunk_index for stable ordering
    def _chunk_index(point: models.Record) -> int:
        payload = point.payload or {}
        return payload.get("chunk_index", 0)

    all_points.sort(key=_chunk_index)

    logger.info("Scrolled %d chunks for doc_id=%s", len(all_points), doc_id)
    return all_points


async def rename_file(doc_id: str, new_filename: str) -> dict[str, Any]:
    """Update the filename payload across all chunks for a given ``doc_id``.

    Uses Qdrant ``set_payload`` with a filter to update all matching points
    in a single call. Returns the count of updated chunks.

    Raises:
        VectorStoreError: If the doc_id is not found or the operation fails.
    """
    settings = get_settings()
    client = await get_qdrant_client()

    # Step 1: Count existing chunks to return in response
    start_ms = now_ms()
    try:
        points, _next_offset = await client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=scroll_rename mode=unexpected doc_id=%s latency_ms=%.1f",
            doc_id,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to query file before rename") from exc

    if len(points) == 0:
        raise VectorStoreError(f"File with doc_id='{doc_id}' not found")

    # Step 2: Count total chunks for this doc_id
    start_ms = now_ms()
    try:
        count_result = await client.count(
            collection_name=settings.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
            exact=True,
        )
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=count_rename mode=unexpected doc_id=%s latency_ms=%.1f",
            doc_id,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to count chunks before rename") from exc

    chunk_count = count_result.count

    # Step 3: Apply set_payload with filter
    points_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
    )

    start_ms = now_ms()
    try:
        await client.set_payload(
            collection_name=settings.collection_name,
            payload={"filename": new_filename},
            points=points_filter,
        )
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=set_payload_rename mode=unexpected doc_id=%s latency_ms=%.1f",
            doc_id,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to update filename in Qdrant") from exc

    logger.info(
        "Renamed file doc_id=%s to '%s' across %d chunks",
        doc_id,
        new_filename,
        chunk_count,
    )
    return {
        "doc_id": doc_id,
        "filename": new_filename,
        "updated_chunks": chunk_count,
    }


async def get_chunk_by_doc_id_and_index(
    doc_id: str, chunk_index: int
) -> dict[str, Any] | None:
    """Fetch a single chunk by doc_id and chunk_index.

    Returns the point payload dict if found, else None.
    """
    settings = get_settings()
    client = await get_qdrant_client()

    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            ),
            models.FieldCondition(
                key="chunk_index",
                match=models.MatchValue(value=chunk_index),
            ),
        ]
    )

    start_ms = now_ms()
    try:
        points, _next_offset = await client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=get_chunk mode=unexpected doc_id=%s chunk_index=%d latency_ms=%.1f",
            doc_id,
            chunk_index,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to retrieve chunk from Qdrant") from exc

    if not points:
        return None

    return {"point_id": str(points[0].id), "payload": points[0].payload or {}}


async def upsert_single_chunk(
    point_id: str,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    payload: dict[str, Any],
) -> None:
    """Upsert a single chunk point with dense and sparse vectors."""
    settings = get_settings()
    client = await get_qdrant_client()

    point = models.PointStruct(
        id=point_id,
        vector={
            "dense": dense_vector,
            "bm25": sparse_vector,
        },
        payload=payload,
    )

    start_ms = now_ms()
    try:
        await client.upsert(
            collection_name=settings.collection_name,
            points=[point],
        )
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=upsert_single_chunk mode=unexpected point_id=%s latency_ms=%.1f",
            point_id,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to upsert chunk to Qdrant") from exc

    logger.info("Upserted single chunk point_id=%s", point_id)


async def delete_single_chunk(point_id: str) -> bool:
    """Delete a single chunk point by its ID.

    Returns True if a point was deleted, False if it didn't exist.
    """
    settings = get_settings()
    client = await get_qdrant_client()

    start_ms = now_ms()
    try:
        await client.delete(
            collection_name=settings.collection_name,
            points=[point_id],
            wait=True,
        )
    except Exception as exc:
        logger.exception(
            "Dependency failure dependency=qdrant operation=delete_single_chunk mode=unexpected point_id=%s latency_ms=%.1f",
            point_id,
            elapsed_ms(start_ms),
        )
        raise VectorStoreError("Failed to delete chunk from Qdrant") from exc

    logger.info("Deleted single chunk point_id=%s", point_id)
    return True


async def close_client() -> None:
    """Close the Qdrant client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
