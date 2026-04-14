"""Docling-based document parser with table stitching."""

from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import SectionHeaderItem, TableItem, TextItem
from docling_core.types.doc.document import DoclingDocument

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── v6: Table Stitching Configuration ────────────────────────────────────────
TABLE_MAX_ROWS_PER_CHUNK = 30


# ─── v6: Table Stitching Helpers ──────────────────────────────────────────────
def _get_md_header(md: str) -> str:
    """Ambil baris pertama (header row) dari markdown table."""
    for line in md.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _col_count(md: str) -> int:
    """Count columns in a markdown table."""
    return _get_md_header(md).count("|")


def _tables_are_continuation(prev_md: str, curr_md: str) -> bool:
    """
    Heuristic: curr adalah continuation dari prev jika:
    (1) header row identik, ATAU
    (2) jumlah kolom sama (dan keduanya punya kolom)
    """
    prev_header = _get_md_header(prev_md)
    curr_header = _get_md_header(curr_md)
    if prev_header and prev_header == curr_header:
        return True
    pc = _col_count(prev_md)
    cc = _col_count(curr_md)
    return bool(pc > 0 and pc == cc)


def _merge_markdown_tables(prev_md: str, curr_md: str) -> str:
    """
    Gabungkan dua markdown table.
    Buang header row + separator row dari curr sebelum concat.
    """
    curr_lines = [line for line in curr_md.strip().splitlines() if line.strip()]
    # Baris 0 = header, baris 1 = separator (---), baris 2+ = data
    body_lines = curr_lines[2:] if len(curr_lines) > 2 else []
    if not body_lines:
        return prev_md
    return prev_md.rstrip() + "\n" + "\n".join(body_lines)


def _split_markdown_table_by_rows(md: str, max_rows: int) -> list[str]:
    """
    Split markdown table yang terlalu besar menjadi beberapa chunk,
    masing-masing tetap menyertakan header + separator row.
    """
    lines = [line for line in md.strip().splitlines() if line.strip()]
    if len(lines) <= 2:
        return [md]
    header = lines[0]
    separator = lines[1]
    data_rows = lines[2:]

    chunks = []
    for i in range(0, len(data_rows), max_rows):
        batch = data_rows[i : i + max_rows]
        chunks.append("\n".join([header, separator, *batch]))
    return chunks if chunks else [md]


def _table_to_markdown(element, doc) -> str | None:
    """
    Export TableItem ke markdown, flatten multi-level header via DataFrame.
    Fallback ke export_to_markdown kalau DataFrame gagal.
    """
    try:
        df = element.export_to_dataframe()
        if df is None or df.empty:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = [
                " > ".join(str(c).strip() for c in col if str(c).strip())
                if isinstance(col, tuple)
                else str(col).strip()
                for col in df.columns
            ]
        else:
            df.columns = [str(c).strip() for c in df.columns]
        df = df.fillna("")
        return df.to_markdown(index=False)
    except Exception:
        pass
    try:
        return element.export_to_markdown(doc=doc)
    except Exception:
        return None


def _prov_page(element) -> int | None:
    """Extract page number from element provenance."""
    try:
        if element.prov:
            return element.prov[0].page_no
    except Exception:
        pass
    return None


def stitch_tables(doc: DoclingDocument) -> list[dict]:
    """
    Walk document body, detect + merge continuation tables.
    Returns list of blocks:
      {"type": "text"|"table", "text": str, "headings": list[str], "page": int|None}
    """
    blocks: list[dict] = []
    current_headings: list[str] = []
    pending_md: str | None = None
    pending_page: int | None = None
    last_page: int | None = None

    def flush_pending():
        nonlocal pending_md, pending_page
        if pending_md:
            parts = _split_markdown_table_by_rows(pending_md, TABLE_MAX_ROWS_PER_CHUNK)
            for part in parts:
                blocks.append(
                    {
                        "type": "table",
                        "text": part,
                        "headings": list(current_headings),
                        "page": pending_page,
                    }
                )
            pending_md = None
            pending_page = None

    for element, _level in doc.iterate_items():
        p = _prov_page(element)
        if p is not None:
            last_page = p

        if isinstance(element, SectionHeaderItem):
            flush_pending()
            current_headings = [element.text.strip()]
            blocks.append(
                {
                    "type": "text",
                    "text": element.text.strip(),
                    "headings": list(current_headings),
                    "page": last_page,
                }
            )

        elif isinstance(element, TableItem):
            md = _table_to_markdown(element, doc)
            if not md or not md.strip():
                continue
            page = _prov_page(element) or last_page

            if pending_md and _tables_are_continuation(pending_md, md):
                pending_md = _merge_markdown_tables(pending_md, md)
            else:
                flush_pending()
                pending_md = md
                pending_page = page

        elif isinstance(element, TextItem):
            flush_pending()
            text = element.text.strip() if element.text else ""
            if text:
                blocks.append(
                    {
                        "type": "text",
                        "text": text,
                        "headings": list(current_headings),
                        "page": last_page,
                    }
                )

    flush_pending()
    logger.info("Table stitching produced %d blocks", len(blocks))
    return blocks


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
