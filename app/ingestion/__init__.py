"""Ingestion pipeline modules."""

# Re-export key symbols — avoid importing from pipeline/upserter here
# to prevent circular imports with app.utils.exceptions.

from app.ingestion.checkpoint import CheckpointManager
from app.ingestion.chunker import Chunk, chunk_document, chunk_document_from_blocks
from app.ingestion.metadata import DocMetadata, extract_doc_metadata
from app.ingestion.normalizer import normalize_table_blocks
from app.ingestion.parser import parse_document, stitch_tables

__all__ = [
    "CheckpointManager",
    "Chunk",
    "DocMetadata",
    "chunk_document",
    "chunk_document_from_blocks",
    "extract_doc_metadata",
    "normalize_table_blocks",
    "parse_document",
    "stitch_tables",
]
