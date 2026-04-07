"""Tests for document ingestion pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.chunker import Chunk
from app.utils.helpers import generate_chunk_point_id


@pytest.mark.asyncio
async def test_upsert_chunks_empty():
    """Upserter handles empty chunk list gracefully."""
    from app.ingestion.upserter import upsert_chunks

    result = await upsert_chunks(chunks=[], doc_id="test", filename="test.pdf")
    assert result == 0


@pytest.mark.asyncio
async def test_upsert_chunks_batch():
    """Upserter embeds and upserts chunks correctly."""
    from app.ingestion.upserter import upsert_chunks

    chunks = [
        Chunk(text="Hello world", headings=["Intro"], chunk_index=0, page=1),
        Chunk(text="Some content", headings=["Body"], chunk_index=1, page=2),
    ]

    with (
        patch("app.ingestion.upserter.embed_svc") as mock_embed,
        patch("app.ingestion.upserter.vs") as mock_vs,
    ):
        mock_embed.embed_texts = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        mock_vs.ensure_collection = AsyncMock()
        mock_vs.delete_points_by_doc_id = AsyncMock()
        mock_vs.upsert_points = AsyncMock()

        result = await upsert_chunks(chunks, doc_id="abc", filename="doc.pdf")

    assert result == 2
    mock_embed.embed_texts.assert_called_once()
    mock_vs.delete_points_by_doc_id.assert_called_once_with("abc")
    mock_vs.upsert_points.assert_called_once()

    call_kwargs = mock_vs.upsert_points.call_args.kwargs
    assert call_kwargs["ids"] == [
        generate_chunk_point_id("abc", 0),
        generate_chunk_point_id("abc", 1),
    ]


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
async def test_pipeline_ingest():
    """Full pipeline runs parse → chunk → upsert."""
    from app.ingestion.pipeline import ingest_document

    mock_doc = MagicMock()
    mock_chunks = [
        Chunk(text="Chunk 1", headings=[], chunk_index=0),
        Chunk(text="Chunk 2", headings=[], chunk_index=1),
    ]

    with (
        patch(
            "app.ingestion.pipeline.parse_document", return_value=mock_doc
        ) as mock_parse,
        patch(
            "app.ingestion.pipeline.chunk_document", return_value=mock_chunks
        ) as mock_chunk,
        patch(
            "app.ingestion.pipeline.upsert_chunks",
            new_callable=AsyncMock,
            return_value=2,
        ) as mock_upsert,
    ):
        result = await ingest_document("test.pdf")

    assert result["filename"] == "test.pdf"
    assert result["chunks_count"] == 2
    mock_parse.assert_called_once()
    mock_chunk.assert_called_once_with(mock_doc)
    mock_upsert.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_chunks_reingestion_replaces_same_doc_points():
    """Repeated ingestion should replace prior document points deterministically."""
    from app.ingestion.upserter import upsert_chunks

    chunks = [
        Chunk(text="Chunk 1", headings=["A"], chunk_index=0, page=1),
        Chunk(text="Chunk 2", headings=["B"], chunk_index=1, page=1),
    ]

    with (
        patch("app.ingestion.upserter.embed_svc") as mock_embed,
        patch("app.ingestion.upserter.vs") as mock_vs,
    ):
        mock_embed.embed_texts = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        mock_vs.ensure_collection = AsyncMock()
        mock_vs.delete_points_by_doc_id = AsyncMock()
        mock_vs.upsert_points = AsyncMock()

        await upsert_chunks(chunks, doc_id="doc-fixed", filename="admin.pdf")
        await upsert_chunks(chunks, doc_id="doc-fixed", filename="admin.pdf")

    # Explicit replacement should run each ingestion call.
    assert mock_vs.delete_points_by_doc_id.call_count == 2

    first_ids = mock_vs.upsert_points.call_args_list[0].kwargs["ids"]
    second_ids = mock_vs.upsert_points.call_args_list[1].kwargs["ids"]
    assert first_ids == second_ids
