"""HybridChunker wrapper and v6 chunking utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── v6 Configuration Constants ────────────────────────────────────────────────
CHUNK_MAX_TOKENS = 768
CHUNK_MIN_TOKENS = 20
QWEN_HF_MODEL = "Qwen/Qwen3-Embedding-8B"

# ─── TOC Chunk Filter Constants ────────────────────────────────────────────────
_TOC_PATTERNS = [
    re.compile(r"\.{4,}"),  # "Bab 1 .......... 3"
    re.compile(r"\b\d+\s*\.\s*\d+\s*\.\s*\d+"),  # "1.2.3" numbering density
]
_TOC_MIN_DOT_RATIO = 0.4  # >40% baris punya dotleader = TOC
_TOC_MIN_LINES = 5


@dataclass
class Chunk:
    """A single text chunk with metadata (v6 enhanced)."""

    text: str
    headings: list[str]
    chunk_index: int
    page: int | None = field(default=None)
    is_table: bool = field(default=False)
    text_raw: str | None = field(default=None)  # original text before LLM normalization


# ─── Qwen Tokenizer (lazy-loaded) ─────────────────────────────────────────────
_qwen_tokenizer = None


def _get_qwen_tokenizer():
    """Lazy-load Qwen tokenizer to avoid import overhead at module load."""
    global _qwen_tokenizer
    if _qwen_tokenizer is None:
        from transformers import AutoTokenizer

        logger.info("Loading Qwen tokenizer: %s…", QWEN_HF_MODEL)
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            QWEN_HF_MODEL, trust_remote_code=True
        )
        logger.info("Qwen tokenizer ready")
    return _qwen_tokenizer


def _token_count(text: str) -> int:
    """Count tokens using Qwen tokenizer."""
    tokenizer = _get_qwen_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


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
    except Exception:
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


# ─── v6: TOC Chunk Filter ─────────────────────────────────────────────────────
def _clean_toc_chunk(text: str) -> str | None:
    """
    Return None kalau chunk terdeteksi sebagai TOC (daftar isi).
    Return text (cleaned) kalau bukan TOC.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= _TOC_MIN_LINES:
        dot_lines = sum(1 for line in lines if _TOC_PATTERNS[0].search(line))
        if dot_lines / len(lines) >= _TOC_MIN_DOT_RATIO:
            return None  # Skip — ini TOC

    # Bersihkan baris kosong berlebih
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    return cleaned if cleaned else None


# ─── v6: Table Detection (fallback untuk text chunks) ─────────────────────────
def _is_table_chunk(text: str) -> bool:
    """Detect if a text chunk looks like a table (fallback for text blocks)."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 2:
        md_lines = sum(
            1 for line in lines if line.startswith("|") and line.endswith("|")
        )
        if md_lines / len(lines) >= 0.6:
            return True
    kv_pattern = re.compile(r"\d+,\s*[\w\s\.\,\(\)]+?\s*\.?\s*=\s*[^.]+\.")
    if len(kv_pattern.findall(text)) >= 3:
        return True
    row_prefix = re.findall(r"\b(\d+),\s*", text)
    return bool(
        len(lines) >= 4
        and len(set(row_prefix)) >= 3
        and len(row_prefix) / len(lines) >= 0.4
    )


# ─── v6: Token-Aware Text Splitter ────────────────────────────────────────────
def _split_text_by_tokens(text: str, max_tokens: int) -> list[str]:
    """
    Naif paragraph-aware splitter untuk text blocks yang > max_tokens.
    Coba split di batas paragraf dulu; kalau masih kegedean, potong keras.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _token_count(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        # Kalau 1 paragraf sendiri sudah > max_tokens, potong per kalimat
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                st = _token_count(sent)
                if current_tokens + st > max_tokens and current:
                    chunks.append("\n\n".join(current))
                    current, current_tokens = [], 0
                current.append(sent)
                current_tokens += st
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


# ─── v6: Block-based Chunking (alternative to chunk_document) ─────────────────
def chunk_document_from_blocks(
    blocks: list[dict],
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> list[Chunk]:
    """
    Convert pre-stitched blocks ke list[Chunk].
    - Table blocks → 1 Chunk per block (sudah di-split oleh stitch_tables)
    - Text blocks  → token-aware split kalau > max_tokens
    """
    chunks: list[Chunk] = []
    idx = 0

    for block in blocks:
        text = block["text"]
        headings = block["headings"]
        page = block.get("page")
        is_table = block["type"] == "table"

        if is_table:
            cleaned = text.strip()
            if not cleaned:
                continue
            if _token_count(cleaned) < CHUNK_MIN_TOKENS:
                continue  # skip noise tabel (header-only, dll)
            chunks.append(
                Chunk(
                    text=cleaned,
                    headings=headings,
                    chunk_index=idx,
                    page=page,
                    is_table=True,
                    text_raw=block.get("text_raw"),
                )
            )
            idx += 1
        else:
            cleaned = _clean_toc_chunk(text)
            if not cleaned:
                continue
            if _token_count(cleaned) < CHUNK_MIN_TOKENS:
                continue  # skip noise teks (cover, tim penyusun, dll)
            # Split kalau terlalu panjang
            parts = (
                _split_text_by_tokens(cleaned, max_tokens)
                if _token_count(cleaned) > max_tokens
                else [cleaned]
            )
            for part in parts:
                if part.strip():
                    chunks.append(
                        Chunk(
                            text=part.strip(),
                            headings=headings,
                            chunk_index=idx,
                            page=page,
                            is_table=_is_table_chunk(part),  # fallback detect
                        )
                    )
                    idx += 1

    logger.info("%d chunks produced (max_tokens=%d)", len(chunks), max_tokens)
    return chunks
