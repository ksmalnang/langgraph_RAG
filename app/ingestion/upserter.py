"""Embed chunks and upsert into Qdrant."""

from __future__ import annotations

from app.ingestion.chunker import Chunk
from app.services import embeddings as embed_svc
from app.services import vectorstore as vs
from app.utils.helpers import generate_point_id
from app.utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 64


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
            return f"{last_heading}\n{chunk.text}"
    return chunk.text


async def upsert_chunks(
    chunks: list[Chunk],
    doc_id: str,
    filename: str,
) -> int:
    """Embed and upsert a list of chunks into Qdrant.

    Returns the number of points upserted.
    """
    if not chunks:
        return 0

    total = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        # Enrich each chunk's text with its last heading for better
        # semantic representation in both dense and sparse vectors.
        enriched_texts = [_enrich_text(c) for c in batch]

        # Embed the batch (dense vectors)
        vectors = await embed_svc.embed_texts(enriched_texts)

        # On the very first batch, make sure the collection exists
        if start == 0:
            await vs.ensure_collection(vector_size=len(vectors[0]))

        # Build point data — keep original text for display, enriched for BM25
        ids = [generate_point_id() for _ in batch]
        payloads = [
            {
                "text": c.text,
                "enriched_text": enriched,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": c.chunk_index,
                "headings": c.headings,
                "page": c.page,
            }
            for c, enriched in zip(batch, enriched_texts)
        ]

        await vs.upsert_points(ids=ids, vectors=vectors, payloads=payloads)
        total += len(batch)

    logger.info("Upserted %d chunks for doc '%s'", total, filename)
    return total
