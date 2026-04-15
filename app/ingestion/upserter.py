"""Embed chunks and upsert into Qdrant (v6 enhanced)."""

from __future__ import annotations

from app.ingestion.chunker import Chunk
from app.ingestion.metadata import extract_doc_metadata
from app.services import embeddings as embed_svc
from app.services import vectorstore as vs
from app.utils.helpers import generate_chunk_point_id
from app.utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 64
SLEEP_BETWEEN_BATCHES = 0.4


def _enrich_text(chunk: Chunk) -> str:
    """Prepend the last heading to the chunk text for richer embeddings.

    When a chunk belongs to a document section, its raw text often lacks the
    heading that gives it context (e.g. "Installation" → "Run pip install …").
    By prepending the most-specific (last) heading we close this semantic gap
    so that both dense and sparse vectors capture the section context.

    Returns the original text unchanged when no headings are available.
    """
    if chunk.headings:
        last_heading = chunk.headings[-1].strip()
        if last_heading:
            return f"[{last_heading}]\n{chunk.text}"
    return chunk.text


async def upsert_chunks(
    chunks: list[Chunk],
    doc_id: str,
    filename: str,
) -> int:
    """Embed and upsert a list of chunks into Qdrant (v6 enhanced).

    Returns the number of points upserted.
    """
    if not chunks:
        return 0

    total = 0
    collection_prepared = False
    num_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    # Extract document metadata once for all chunks
    doc_meta = await extract_doc_metadata(filename)
    logger.info(
        "Metadata: category=%s, year=%s", doc_meta.doc_category, doc_meta.academic_year
    )

    for batch_num, start in enumerate(range(0, len(chunks), BATCH_SIZE), 1):
        batch = chunks[start : start + BATCH_SIZE]
        logger.info("Batch %d/%d — %d chunks", batch_num, num_batches, len(batch))

        # Enrich each chunk's text with its last heading for better
        # semantic representation in both dense and sparse vectors.
        enriched_texts = [_enrich_text(c) for c in batch]

        # Embed the batch (dense vectors)
        vectors = await embed_svc.embed_texts(enriched_texts)

        # Ensure the collection exists, then replace any older points
        # for the same doc_id so re-ingestion stays deterministic.
        if not collection_prepared:
            await vs.ensure_collection(vector_size=len(vectors[0]))
            await vs.delete_points_by_doc_id(doc_id)
            collection_prepared = True

        # Build point data — keep original text for display, enriched for BM25
        ids = [generate_chunk_point_id(doc_id, c.chunk_index) for c in batch]
        payloads = [
            {
                "text": c.text,
                "enriched_text": enriched,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": c.chunk_index,
                "headings": c.headings,
                "page": c.page,
                "doc_category": doc_meta.doc_category,
                "academic_year": doc_meta.academic_year,
                "content_type": "table" if c.is_table else "text",
                **({"text_raw": c.text_raw} if c.text_raw else {}),
            }
            for c, enriched in zip(batch, enriched_texts, strict=False)
        ]

        await vs.upsert_points(ids=ids, vectors=vectors, payloads=payloads)
        total += len(batch)
        logger.info("Upserted %d/%d chunks", total, len(chunks))

        if start + BATCH_SIZE < len(chunks):
            logger.debug("Sleeping %.1fs…", SLEEP_BETWEEN_BATCHES)
            import asyncio

            await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

    logger.info("Done — %d chunks for '%s'", total, filename)
    return total
