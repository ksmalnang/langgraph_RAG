"""HybridChunker wrapper for splitting documents into chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    text: str
    headings: list[str]
    chunk_index: int
    page: int | None = field(default=None)


def _extract_page(rc) -> int | None:
    """Extract the first page number from a Docling chunk's provenance metadata.

    Docling stores page provenance in ``chunk.meta.doc_items[*].prov[*].page_no``.
    Returns the smallest page number found, or ``None`` when unavailable.
    """
    try:
        if not (hasattr(rc, "meta") and rc.meta and hasattr(rc.meta, "doc_items")):
            return None
        pages: list[int] = []
        for item in rc.meta.doc_items:
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no") and prov.page_no is not None:
                        pages.append(prov.page_no)
        return min(pages) if pages else None
    except Exception:  # noqa: BLE001 — defensive; never fail the pipeline
        return None


def chunk_document(doc: DoclingDocument) -> list[Chunk]:
    """Split a DoclingDocument into sized text chunks.

    Uses Docling's HybridChunker for structure-aware splitting.
    """
    settings = get_settings()
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=settings.chunk_max_tokens,
        merge_peers=True,
    )

    raw_chunks = list(chunker.chunk(doc))
    chunks: list[Chunk] = []

    for idx, rc in enumerate(raw_chunks):
        text = rc.text.strip()
        if not text:
            continue

        headings: list[str] = []
        if hasattr(rc, "meta") and rc.meta and hasattr(rc.meta, "headings"):
            headings = list(rc.meta.headings) if rc.meta.headings else []

        page = _extract_page(rc)

        chunks.append(Chunk(text=text, headings=headings, chunk_index=idx, page=page))

    logger.info(
        "Chunked document into %d chunks (max_tokens=%d)",
        len(chunks),
        settings.chunk_max_tokens,
    )
    return chunks
