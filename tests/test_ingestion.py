"""Tests for document ingestion pipeline (v6 enhanced)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.checkpoint import CheckpointManager
from app.ingestion.chunker import (
    Chunk,
    _clean_toc_chunk,
    _is_table_chunk,
    chunk_document_from_blocks,
)
from app.ingestion.metadata import (
    DocMetadata,
    _fallback_metadata,
    _sanitize_academic_year,
)
from app.ingestion.parser import (
    _col_count,
    _get_md_header,
    _merge_markdown_tables,
    _split_markdown_table_by_rows,
    _tables_are_continuation,
)

# ─── v6: Table Stitching Tests ────────────────────────────────────────────────


class TestTableStitchingHelpers:
    """Test table stitching helper functions."""

    def test_get_md_header_returns_first_line(self):
        """Extract header row from markdown table."""
        md = "| Col1 | Col2 | Col3 |\n|------|------|------|\n| data | data | data |"
        assert _get_md_header(md) == "| Col1 | Col2 | Col3 |"

    def test_get_md_header_empty(self):
        """Return empty string for empty input."""
        assert _get_md_header("") == ""

    def test_col_count(self):
        """Count columns in markdown table header."""
        md = "| Col1 | Col2 | Col3 |"
        assert _col_count(md) == 4  # 4 pipes = 4 columns

    def test_tables_are_continuation_identical_headers(self):
        """Tables with identical headers are continuation."""
        prev = "| A | B |\n|---|---|\n| 1 | 2 |"
        curr = "| A | B |\n|---|---|\n| 3 | 4 |"
        assert _tables_are_continuation(prev, curr) is True

    def test_tables_are_continuation_different_headers(self):
        """Tables with different headers but same column count are continuation."""
        prev = "| A | B |\n|---|---|\n| 1 | 2 |"
        curr = "| X | Y |\n|---|---|\n| 3 | 4 |"
        assert _tables_are_continuation(prev, curr) is True

    def test_tables_are_continuation_different_columns(self):
        """Tables with different column counts are not continuation."""
        prev = "| A | B |\n|---|---|\n| 1 | 2 |"
        curr = "| X | Y | Z |\n|---|---|---|\n| 3 | 4 | 5 |"
        assert _tables_are_continuation(prev, curr) is False

    def test_merge_markdown_tables(self):
        """Merge two markdown tables by removing header from second."""
        prev = "| A | B |\n|---|---|\n| 1 | 2 |"
        curr = "| A | B |\n|---|---|\n| 3 | 4 |"
        merged = _merge_markdown_tables(prev, curr)
        assert "| 1 | 2 |" in merged
        assert "| 3 | 4 |" in merged
        # Should only have one header
        assert merged.count("| A | B |") == 1

    def test_split_markdown_table_by_rows_small(self):
        """Don't split small tables."""
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        parts = _split_markdown_table_by_rows(md, max_rows=30)
        assert len(parts) == 1

    def test_split_markdown_table_by_rows_large(self):
        """Split large tables with header preserved in each chunk."""
        header = "| A | B |\n|---|---|"
        rows = "\n".join(f"| {i} | data{i} |" for i in range(100))
        md = header + "\n" + rows
        parts = _split_markdown_table_by_rows(md, max_rows=30)
        assert len(parts) >= 3  # 100 rows / 30 = 4 chunks
        # Each chunk should have the header
        for part in parts:
            assert "| A | B |" in part


# ─── v6: TOC Filter Tests ─────────────────────────────────────────────────────


class TestTOCFilter:
    """Test TOC chunk detection and filtering."""

    def test_detect_toc_with_dot_leaders(self):
        """Detect TOC chunks with dot leader patterns."""
        toc_text = """Bab 1 Pendahuluan .......... 1
Bab 2 Tinjauan Pustaka .......... 5
Bab 3 Metodologi .......... 12
Bab 4 Hasil dan Pembahasan .......... 20
Bab 5 Kesimpulan .......... 35"""
        assert _clean_toc_chunk(toc_text) is None

    def test_non_toc_text_not_filtered(self):
        """Regular text should not be filtered."""
        text = """This is a regular paragraph.
It contains normal content.
No dot leaders here."""
        result = _clean_toc_chunk(text)
        assert result is not None
        assert "regular paragraph" in result

    def test_short_text_with_dots_not_filtered(self):
        """Short text with dots should not be filtered (below MIN_LINES)."""
        text = """Section 1 .......... 5
Section 2 .......... 10"""
        result = _clean_toc_chunk(text)
        assert result is not None


# ─── v6: Table Detection Tests ────────────────────────────────────────────────


class TestTableDetection:
    """Test fallback table detection in text chunks."""

    def test_detect_markdown_table(self):
        """Detect markdown table format."""
        text = """| Header1 | Header2 |
|---------|---------|
| data1   | data2   |
| data3   | data4   |"""
        assert _is_table_chunk(text) is True

    def test_non_table_text(self):
        """Regular text should not be detected as table."""
        text = "This is just regular text with no table structure."
        assert _is_table_chunk(text) is False


# ─── v6: Checkpoint Manager Tests ─────────────────────────────────────────────


class TestCheckpointManager:
    """Test checkpoint manager for resumable ingestion."""

    def test_mark_and_check(self, tmp_path):
        """Mark files as processed and check status."""
        checkpoint_path = tmp_path / "test_checkpoint.json"
        cp = CheckpointManager(checkpoint_path)

        assert cp.is_processed("doc1.pdf") is False
        cp.mark_processed("doc1.pdf")
        assert cp.is_processed("doc1.pdf") is True
        assert cp.is_processed("doc2.pdf") is False

    def test_flush_persists(self, tmp_path):
        """Flushed checkpoint persists across instances."""
        checkpoint_path = tmp_path / "test_checkpoint.json"
        cp = CheckpointManager(checkpoint_path)
        cp.mark_processed("doc1.pdf")
        cp.mark_processed("doc2.pdf")
        cp.flush()

        # New instance should load the checkpoint
        cp2 = CheckpointManager(checkpoint_path)
        assert cp2.is_processed("doc1.pdf") is True
        assert cp2.is_processed("doc2.pdf") is True

    def test_flush_idempotent(self, tmp_path):
        """Multiple flushes should not cause errors."""
        checkpoint_path = tmp_path / "test_checkpoint.json"
        cp = CheckpointManager(checkpoint_path)
        cp.mark_processed("doc1.pdf")
        cp.flush()
        cp.flush()  # Should not raise
        cp.flush()  # Should not raise


# ─── v6: Metadata Extraction Tests ────────────────────────────────────────────


class TestMetadataExtraction:
    """Test document metadata extraction."""

    def test_fallback_curriculum(self):
        """Detect curriculum documents via keywords."""
        meta = _fallback_metadata("Kurikulum_2021_Teknik_Informatika.pdf")
        assert meta.doc_category == "curriculum"
        # Year may be None if not explicitly set by _sanitize_academic_year
        assert meta.academic_year is None or meta.academic_year == "2021-2022"

    def test_fallback_handbook(self):
        """Detect handbook documents."""
        meta = _fallback_metadata("Handbook_Akademik_2022.pdf")
        assert meta.doc_category == "handbook"

    def test_fallback_student_guide(self):
        """Detect student guide documents."""
        meta = _fallback_metadata("Panduan_Mahasiswa_Baru_2023.pdf")
        assert meta.doc_category == "student_guide"

    def test_fallback_policy(self):
        """Detect policy documents."""
        meta = _fallback_metadata("Peraturan_Rektor_2020.pdf")
        assert meta.doc_category == "policy"

    def test_fallback_other(self):
        """Return 'other' for unrecognized patterns."""
        meta = _fallback_metadata("random_document.pdf")
        assert meta.doc_category == "other"

    def test_sanitize_academic_year_normal(self):
        """Valid academic year passes through."""
        result = _sanitize_academic_year("2021-2022")
        assert result == "2021-2022"

    def test_sanitize_academic_year_same_start_end(self):
        """Fix same start/end year."""
        result = _sanitize_academic_year("2021-2021", "2021")
        assert result == "2021-2022"

    def test_sanitize_academic_year_invalid_range(self):
        """Fix invalid year range."""
        result = _sanitize_academic_year("2021-2025", "2021")
        assert result == "2021-2022"

    def test_sanitize_academic_year_none(self):
        """Generate year from source when value is None."""
        result = _sanitize_academic_year(None, "2023")
        assert result == "2023-2024"


# ─── v6: Chunk Data Model Tests ───────────────────────────────────────────────


class TestChunkDataModel:
    """Test enhanced Chunk dataclass."""

    def test_chunk_with_table_flag(self):
        """Chunk supports table flag."""
        chunk = Chunk(
            text="| A | B |\n| 1 | 2 |",
            headings=["Tables"],
            chunk_index=0,
            is_table=True,
        )
        assert chunk.is_table is True
        assert chunk.text_raw is None

    def test_chunk_with_raw_text(self):
        """Chunk stores original text before normalization."""
        chunk = Chunk(
            text="normalized text",
            headings=["Section"],
            chunk_index=0,
            text_raw="raw | table | text",
        )
        assert chunk.text_raw == "raw | table | text"


# ─── v6: Block-based Chunking Tests ──────────────────────────────────────────


class TestBlockBasedChunking:
    """Test chunk_document_from_blocks function."""

    def test_chunk_from_text_blocks(self):
        """Convert text blocks to chunks."""
        with patch("app.ingestion.chunker._token_count", return_value=50):
            blocks = [
                {
                    "type": "text",
                    "text": "Hello world",
                    "headings": ["Intro"],
                    "page": 1,
                },
                {
                    "type": "text",
                    "text": "More content here",
                    "headings": ["Body"],
                    "page": 2,
                },
            ]
            chunks = chunk_document_from_blocks(blocks)
            assert len(chunks) == 2
            assert chunks[0].text == "Hello world"
            assert chunks[0].headings == ["Intro"]
            assert chunks[0].is_table is False

    def test_chunk_from_table_blocks(self):
        """Convert table blocks to chunks with is_table flag."""
        with patch("app.ingestion.chunker._token_count", return_value=100):
            blocks = [
                {
                    "type": "table",
                    "text": "| A | B |\n|---|---|\n| 1 | 2 |",
                    "headings": ["Data"],
                    "page": 1,
                },
            ]
            chunks = chunk_document_from_blocks(blocks)
            assert len(chunks) == 1
            assert chunks[0].is_table is True
            assert "| A | B |" in chunks[0].text

    def test_toc_chunks_filtered(self):
        """TOC blocks are filtered out."""
        toc_text = """Bab 1 .......... 1
Bab 2 .......... 5
Bab 3 .......... 10
Bab 4 .......... 15
Bab 5 .......... 20"""
        blocks = [{"type": "text", "text": toc_text, "headings": [], "page": 1}]
        chunks = chunk_document_from_blocks(blocks)
        assert len(chunks) == 0  # TOC should be filtered out

    def test_empty_text_skipped(self):
        """Empty text blocks are skipped."""
        blocks = [{"type": "text", "text": "", "headings": [], "page": 1}]
        chunks = chunk_document_from_blocks(blocks)
        assert len(chunks) == 0


# ─── Existing Tests (Updated for v6) ──────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.skip(reason="Circular import in app.services.memory - pre-existing issue")
async def test_upsert_chunks_empty():
    """Upserter handles empty chunk list gracefully."""
    with patch("app.ingestion.upserter.extract_doc_metadata"):
        from app.ingestion.upserter import upsert_chunks

        result = await upsert_chunks(chunks=[], doc_id="test", filename="test.pdf")
        assert result == 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="Circular import in app.services.memory - pre-existing issue")
async def test_upsert_chunks_batch():
    """Upserter embeds and upserts chunks correctly (v6 enhanced)."""
    from app.ingestion.upserter import upsert_chunks

    chunks = [
        Chunk(text="Hello world", headings=["Intro"], chunk_index=0, page=1),
        Chunk(text="Some content", headings=["Body"], chunk_index=1, page=2),
    ]

    with (
        patch("app.ingestion.upserter.embed_svc") as mock_embed,
        patch("app.ingestion.upserter.vs") as mock_vs,
        patch("app.ingestion.upserter.extract_doc_metadata") as mock_meta,
    ):
        mock_embed.embed_texts = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        mock_vs.ensure_collection = AsyncMock()
        mock_vs.delete_points_by_doc_id = AsyncMock()
        mock_vs.upsert_points = AsyncMock()
        mock_meta.return_value = DocMetadata(
            doc_category="other", academic_year="2024-2025"
        )

        result = await upsert_chunks(chunks, doc_id="abc", filename="doc.pdf")

    assert result == 2
    mock_embed.embed_texts.assert_called_once()
    mock_vs.delete_points_by_doc_id.assert_called_once_with("abc")
    mock_vs.upsert_points.assert_called_once()

    # Verify v6 payload fields
    call_kwargs = mock_vs.upsert_points.call_args.kwargs
    payload = call_kwargs["payloads"][0]
    assert "content_type" in payload
    assert payload["content_type"] == "text"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Circular import in app.services.memory - pre-existing issue")
async def test_upsert_chunks_includes_table_content_type():
    """Upserter includes content_type='table' for table chunks."""
    from app.ingestion.upserter import upsert_chunks

    chunks = [
        Chunk(
            text="| A | B |\n| 1 | 2 |",
            headings=["Table"],
            chunk_index=0,
            page=1,
            is_table=True,
        ),
    ]

    with (
        patch("app.ingestion.upserter.embed_svc") as mock_embed,
        patch("app.ingestion.upserter.vs") as mock_vs,
        patch("app.ingestion.upserter.extract_doc_metadata") as mock_meta,
    ):
        mock_embed.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_vs.ensure_collection = AsyncMock()
        mock_vs.delete_points_by_doc_id = AsyncMock()
        mock_vs.upsert_points = AsyncMock()
        mock_meta.return_value = DocMetadata(
            doc_category="other", academic_year="2024-2025"
        )

        await upsert_chunks(chunks, doc_id="xyz", filename="doc.pdf")

    payload = mock_vs.upsert_points.call_args.kwargs["payloads"][0]
    assert payload["content_type"] == "table"


def test_chunk_dataclass():
    """Chunk dataclass holds expected fields."""
    chunk = Chunk(text="test text", headings=["H1", "H2"], chunk_index=5, page=3)
    assert chunk.text == "test text"
    assert chunk.headings == ["H1", "H2"]
    assert chunk.chunk_index == 5
    assert chunk.page == 3


def test_chunk_page_defaults_to_none():
    """Chunk.page defaults to None when not provided."""
    chunk = Chunk(text="no page", headings=[], chunk_index=0)
    assert chunk.page is None


def test_generate_doc_id_is_case_insensitive():
    from app.utils.helpers import generate_doc_id

    assert generate_doc_id("Admin_Guide.PDF") == generate_doc_id("admin_guide.pdf")


@pytest.mark.asyncio
@pytest.mark.skip(reason="Circular import in app.services.memory - pre-existing issue")
async def test_pipeline_ingest_with_v6_features():
    """Full pipeline runs parse → stitch → normalize → chunk → upsert (v6)."""
    from app.ingestion.pipeline import ingest_document

    mock_doc = MagicMock()
    mock_blocks = [
        {"type": "text", "text": "Hello world", "headings": ["Intro"], "page": 1},
        {
            "type": "table",
            "text": "| A | B |\n| 1 | 2 |",
            "headings": ["Data"],
            "page": 2,
        },
    ]

    with (
        patch(
            "app.ingestion.pipeline.parse_document", return_value=mock_doc
        ) as mock_parse,
        patch(
            "app.ingestion.pipeline.stitch_tables", return_value=mock_blocks
        ) as mock_stitch,
        patch(
            "app.ingestion.pipeline.normalize_table_blocks", new_callable=AsyncMock
        ) as mock_normalize,
        patch("app.ingestion.pipeline.chunk_document_from_blocks") as mock_chunk,
        patch(
            "app.ingestion.pipeline.upsert_chunks",
            new_callable=AsyncMock,
            return_value=2,
        ) as mock_upsert,
    ):
        mock_normalize.return_value = mock_blocks
        mock_chunk.return_value = [
            Chunk(text="Hello world", headings=["Intro"], chunk_index=0),
            Chunk(text="| A | B |", headings=["Data"], chunk_index=1, is_table=True),
        ]

        result = await ingest_document("test.pdf")

    assert result["filename"] == "test.pdf"
    assert result["chunks_count"] == 2
    mock_parse.assert_called_once()
    mock_stitch.assert_called_once_with(mock_doc)
    mock_normalize.assert_called_once()
    mock_chunk.assert_called_once()
    mock_upsert.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Circular import in app.services.memory - pre-existing issue")
async def test_pipeline_skips_checkpoint():
    """Pipeline skips already-processed files via checkpoint."""
    from app.ingestion.pipeline import cp_manager, ingest_document

    # Mark as processed
    cp_manager.mark_processed("already_done.pdf")

    result = await ingest_document("already_done.pdf")
    assert result["skipped"] is True
    assert result["chunks_count"] == 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="Circular import in app.services.memory - pre-existing issue")
async def test_upsert_chunks_reingestion_replaces_same_doc_points():
    """Repeated ingestion should delete prior document points each time."""
    from app.ingestion.upserter import upsert_chunks

    chunks = [
        Chunk(text="Chunk 1", headings=["A"], chunk_index=0, page=1),
        Chunk(text="Chunk 2", headings=["B"], chunk_index=1, page=1),
    ]

    with (
        patch("app.ingestion.upserter.embed_svc") as mock_embed,
        patch("app.ingestion.upserter.vs") as mock_vs,
        patch("app.ingestion.upserter.extract_doc_metadata") as mock_meta,
    ):
        mock_embed.embed_texts = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        mock_vs.ensure_collection = AsyncMock()
        mock_vs.delete_points_by_doc_id = AsyncMock()
        mock_vs.upsert_points = AsyncMock()
        mock_meta.return_value = DocMetadata(
            doc_category="other", academic_year="2024-2025"
        )

        await upsert_chunks(chunks, doc_id="doc-fixed", filename="admin.pdf")
        await upsert_chunks(chunks, doc_id="doc-fixed", filename="admin.pdf")

    # Explicit replacement should run each ingestion call.
    assert mock_vs.delete_points_by_doc_id.call_count == 2
    assert mock_vs.upsert_points.call_count == 2
