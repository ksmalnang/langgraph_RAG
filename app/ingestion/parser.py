"""Docling-based document parser."""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from app.utils.logger import get_logger

logger = get_logger(__name__)


def parse_document(file_path: str | Path) -> DoclingDocument:
    """Convert a document file to a DoclingDocument.

    Supports PDF, DOCX, PPTX, HTML, and other Docling-supported formats.
    """
    file_path = Path(file_path)
    logger.info("Parsing document: %s", file_path.name)

    # Process halaman per halaman atau per 10 halaman
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        page_batch_size=1,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(file_path))

    logger.info("Parsed '%s' successfully", file_path.name)
    return result.document
