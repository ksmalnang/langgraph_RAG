"""Main ingestion orchestrator: parse → chunk → embed → upsert."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.ingestion.parser import parse_document
from app.ingestion.upserter import upsert_chunks
from app.utils.helpers import generate_doc_id
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def ingest_document(file_path: str | Path) -> dict:
    """Run the full ingestion pipeline for a single document.

    Returns a summary dict with ``doc_id``, ``filename``, and ``chunks_count``.
    """
    file_path = Path(file_path)
    filename = file_path.name
    doc_id = generate_doc_id(filename)

    logger.info("Starting ingestion for '%s' (doc_id=%s)", filename, doc_id)

    # 1. Parse
    doc = parse_document(file_path)

    # 2. Chunk
    chunks = chunk_document(doc)

    if not chunks:
        logger.warning("No chunks produced for '%s'", filename)
        return {"doc_id": doc_id, "filename": filename, "chunks_count": 0}

    # 3. Embed + Upsert
    count = await upsert_chunks(chunks, doc_id=doc_id, filename=filename)

    logger.info("Ingestion complete: '%s' → %d chunks", filename, count)
    return {"doc_id": doc_id, "filename": filename, "chunks_count": count}
