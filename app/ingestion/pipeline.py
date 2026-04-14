"""Main ingestion orchestrator: parse → stitch → normalize → chunk → embed → upsert."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

from app.ingestion.checkpoint import CheckpointManager
from app.ingestion.chunker import chunk_document_from_blocks
from app.ingestion.normalizer import normalize_table_blocks
from app.ingestion.parser import parse_document, stitch_tables
from app.ingestion.upserter import upsert_chunks
from app.utils.helpers import generate_doc_id
from app.utils.logger import get_logger

logger = get_logger(__name__)

# v6: Checkpoint manager (module-level singleton)
cp_manager = CheckpointManager()

# v6: Semaphore for concurrent document ingestion
MAX_CONCURRENT_DOCS = 2
_doc_sem = asyncio.Semaphore(MAX_CONCURRENT_DOCS)


async def ingest_document(file_path: str | Path) -> dict:
    """Run the full ingestion pipeline for a single document (v6 enhanced).

    Returns a summary dict with ``doc_id``, ``filename``, and ``chunks_count``.
    """
    file_path = Path(file_path)
    filename = file_path.name

    # v6: Checkpoint check
    if cp_manager.is_processed(filename):
        logger.info("Skipping '%s' (checkpointed)", filename)
        return {"doc_id": "", "filename": filename, "chunks_count": 0, "skipped": True}

    doc_id = generate_doc_id(filename)

    logger.info("Starting ingestion for '%s' (doc_id=%s)", filename, doc_id)
    t0 = time.time()

    try:
        # 1. Parse
        doc = parse_document(file_path)

        # 2. v6: Stitch tables (merge multi-page tables)
        blocks = stitch_tables(doc)
        n_tables = sum(1 for b in blocks if b["type"] == "table")
        n_texts = sum(1 for b in blocks if b["type"] == "text")
        logger.info(
            "%d blocks after stitching (%d tables, %d text)",
            len(blocks),
            n_tables,
            n_texts,
        )

        # 3. v6: Normalize table blocks via LLM
        blocks = await normalize_table_blocks(blocks)

        # 4. v6: Chunk from blocks (TOC filter, token-aware split)
        chunks = chunk_document_from_blocks(blocks)

        if not chunks:
            logger.warning("No chunks produced for '%s'", filename)
            return {"doc_id": doc_id, "filename": filename, "chunks_count": 0}

        # 5. Embed + Upsert (with content_type, metadata)
        count = await upsert_chunks(chunks, doc_id=doc_id, filename=filename)

        # v6: Mark as processed in checkpoint
        cp_manager.mark_processed(filename)

        elapsed = time.time() - t0
        logger.info(
            "Ingestion complete: '%s' → %d chunks (%.1fs)", filename, count, elapsed
        )
        return {"doc_id": doc_id, "filename": filename, "chunks_count": count}

    except Exception as e:
        logger.exception("Failed '%s': %s", filename, e)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": 0,
            "error": str(e),
        }


async def ingest_document_with_semaphore(file_path: str | Path) -> dict:
    """Wrap ingest_document with semaphore for concurrency control."""
    async with _doc_sem:
        return await ingest_document(file_path)


def flush_checkpoint() -> None:
    """Flush checkpoint to disk."""
    cp_manager.flush()
