"""LLM-based table normalization for improved RAG retrieval."""

from __future__ import annotations

import asyncio

import httpx

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── v6: LLM Normalization Configuration ──────────────────────────────────────
LLM_NORMALIZE_MODEL = "openrouter/elephant-alpha"
LLM_NORMALIZE_CONCURRENCY = 8  # max concurrent LLM calls
LLM_NORMALIZE_TIMEOUT = 30.0  # seconds per call
LLM_NORMALIZE_MAX_TOKENS = 1024

# Semaphore for controlling concurrency
_normalize_sem = asyncio.Semaphore(LLM_NORMALIZE_CONCURRENCY)

# ─── v6: Prompt Templates ─────────────────────────────────────────────────────
_NORMALIZE_SYSTEM = """Kamu adalah data cleaner untuk sistem RAG (Retrieval-Augmented Generation) dokumen akademik universitas.
Tugasmu: normalisasi teks tabel menjadi teks terstruktur yang mudah dibaca dan di-retrieve oleh LLM.

ATURAN:
- Pertahankan SEMUA informasi. Jangan hilangkan data apapun.
- Ubah format noisy (misalnya "6, KodeMK. = IF21W0106.") menjadi teks yang natural dan terstruktur.
- Gunakan format yang konsisten antar entri.
- Jangan tambahkan informasi yang tidak ada di input.
- Output HANYA teks hasil normalisasi, tanpa preamble, tanpa penjelasan."""

_NORMALIZE_USER_TMPL = """Berikut adalah teks tabel dari dokumen administrasi universitas. Normalisasi menjadi teks terstruktur yang bersih:

{table_text}
"""


async def _normalize_one(table_text: str, heading: str, idx: int) -> str:
    """
    Kirim 1 table chunk ke LLM untuk dinormalisasi.
    Return teks hasil normalisasi, atau teks asli kalau LLM gagal.
    """
    settings = get_settings()
    async with _normalize_sem:
        try:
            async with httpx.AsyncClient(timeout=LLM_NORMALIZE_TIMEOUT) as client:
                resp = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    json={
                        "model": LLM_NORMALIZE_MODEL,
                        "messages": [
                            {"role": "system", "content": _NORMALIZE_SYSTEM},
                            {
                                "role": "user",
                                "content": _NORMALIZE_USER_TMPL.format(
                                    table_text=table_text
                                ),
                            },
                        ],
                        "temperature": 0.0,
                        "max_tokens": LLM_NORMALIZE_MAX_TOKENS,
                    },
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                result = resp.json()["choices"][0]["message"]["content"].strip()
                if not result:
                    raise ValueError("Empty response from LLM")
                return result
        except Exception as e:
            logger.warning(
                "Normalize failed for table chunk %d [%s]: %s", idx, heading[:40], e
            )
            return table_text  # fallback: gunakan teks asli


def _split_table_block_for_normalize(block: dict, max_rows: int = 30) -> list[dict]:
    """
    Split table block yang terlalu besar secara deterministik
    sebelum dikirim ke LLM. Setiap sub-block tetap punya heading yang sama.
    Catatan: stitch_tables sudah split, jadi ini safety net saja.
    """
    from app.ingestion.parser import _split_markdown_table_by_rows

    parts = _split_markdown_table_by_rows(block["text"], max_rows)
    if len(parts) == 1:
        return [block]
    return [{**block, "text": part} for part in parts]


async def normalize_table_blocks(blocks: list[dict], max_rows: int = 30) -> list[dict]:
    """
    Normalisasi semua table blocks via LLM secara paralel.
    Text blocks dibiarkan apa adanya.

    Alur:
    1. Pisahkan table vs non-table blocks, pertahankan urutan via index
    2. Split table blocks yang terlalu besar (deterministik)
    3. asyncio.gather semua LLM calls (dibatasi semaphore)
    4. Rebuild list blocks dengan urutan yang benar
    """
    # Expand table blocks yang perlu di-split, pertahankan posisi
    expanded: list[dict] = []
    for block in blocks:
        if block["type"] == "table":
            expanded.extend(_split_table_block_for_normalize(block, max_rows))
        else:
            expanded.append(block)

    # Kumpulkan index table blocks
    table_indices = [i for i, b in enumerate(expanded) if b["type"] == "table"]
    table_blocks = [expanded[i] for i in table_indices]

    if not table_blocks:
        return expanded

    logger.info(
        "Normalizing %d table chunks (concurrency=%d)…",
        len(table_blocks),
        LLM_NORMALIZE_CONCURRENCY,
    )

    # Buat coroutine untuk setiap table block
    tasks = [
        _normalize_one(
            table_text=b["text"],
            heading=(b["headings"][-1] if b["headings"] else ""),
            idx=i,
        )
        for i, b in enumerate(table_blocks)
    ]

    # Jalankan semua sekaligus — semaphore yang batasi concurrency
    normalized_texts = await asyncio.gather(*tasks)

    # Tulis hasil normalisasi kembali ke expanded list
    for list_idx, norm_text in zip(table_indices, normalized_texts, strict=False):
        expanded[list_idx] = {
            **expanded[list_idx],
            "text": norm_text,
            "text_raw": expanded[list_idx]["text"],  # simpan asli untuk debug
        }

    success = sum(
        1
        for orig, norm in zip(table_blocks, normalized_texts, strict=False)
        if norm != orig["text"]
    )
    logger.info(
        "Normalized %d/%d table chunks (%d used fallback)",
        success,
        len(table_blocks),
        len(table_blocks) - success,
    )
    return expanded
