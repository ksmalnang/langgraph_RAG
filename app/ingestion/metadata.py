"""Enhanced metadata extraction with regex fallback and LLM support."""

from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── v6: Metadata Model ───────────────────────────────────────────────────────
class DocMetadata(BaseModel):
    """Document metadata extracted from filename."""

    doc_category: str = Field(
        description="handbook, curriculum, student_guide, policy, other"
    )
    academic_year: str | None = Field(description="YYYY-YYYY format or null")


# ─── v6: Keyword Map for Regex Fallback ───────────────────────────────────────
_CAT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("curriculum", ["kurikulum", "curriculum", "mata kuliah", "rps", "cpl", "silabus"]),
    ("handbook", ["handbook", "buku panduan", "panduan akademik", "pedoman"]),
    ("student_guide", ["mahasiswa", "student", "kemahasiswaan", "registrasi"]),
    ("policy", ["kebijakan", "peraturan", "sk rektor", "statuta", "policy"]),
]


# ─── v6: Metadata Cache ───────────────────────────────────────────────────────
_meta_cache: dict[str, DocMetadata] = {}


def _sanitize_academic_year(
    value: str | None, source_year: str | None = None
) -> str | None:
    """Validasi academic_year, fix kalau LLM return "2021-2021" atau format lain yang salah."""
    if not value:
        return f"{source_year}-{int(source_year) + 1}" if source_year else None
    m = re.match(r"(20\d{2})-(20\d{2})", value)
    if not m:
        return f"{source_year}-{int(source_year) + 1}" if source_year else None
    start, end = int(m.group(1)), int(m.group(2))
    if end - start != 1:
        return f"{start}-{start + 1}"
    return value


def _fallback_metadata(filename: str) -> DocMetadata:
    """Extract metadata via regex patterns (fast fallback)."""
    name_lower = filename.lower()
    year_match = re.search(r"\b(20\d{2})\b", filename)
    year = year_match.group(1) if year_match else None
    academic_year = _sanitize_academic_year(None, year)

    for cat, keywords in _CAT_KEYWORDS:
        if any(k in name_lower for k in keywords):
            return DocMetadata(doc_category=cat, academic_year=academic_year)
    return DocMetadata(doc_category="other", academic_year=academic_year)


async def extract_doc_metadata(filename: str) -> DocMetadata:
    """
    Extract document metadata from filename.
    Uses regex fallback first, then LLM if regex returns "other".
    """
    if filename in _meta_cache:
        return _meta_cache[filename]

    # Regex fallback dulu — cukup andal untuk pola filename kampus
    fallback = _fallback_metadata(filename)
    if fallback.doc_category != "other":
        logger.info(
            "Metadata via regex: category=%s, year=%s",
            fallback.doc_category,
            fallback.academic_year,
        )
        _meta_cache[filename] = fallback
        return fallback

    # Kalau "other", coba LLM
    year_match = re.search(r"\b(20\d{2})\b", filename)
    source_year = year_match.group(1) if year_match else None
    settings = get_settings()

    prompt = (
        "Extract structured metadata from this Indonesian university document filename.\n"
        "Return ONLY valid JSON with keys: doc_category, academic_year.\n"
        "doc_category must be one of: handbook, curriculum, student_guide, policy, other.\n"
        "academic_year format: YYYY-YYYY (e.g. 2021-2022, end year must equal start year + 1) or null.\n\n"
        f'Filename: "{filename}"'
    )
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                json={
                    "model": settings.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                    "max_tokens": 128,
                    "provider": {"order": ["stealth"], "allow_fallbacks": False},
                },
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"]
            raw = (
                raw_content.strip().removeprefix("```json").removesuffix("```").strip()
            )
            meta = DocMetadata.model_validate_json(raw)
            meta = DocMetadata(
                doc_category=meta.doc_category,
                academic_year=_sanitize_academic_year(meta.academic_year, source_year),
            )
            _meta_cache[filename] = meta
            logger.info(
                "Metadata via LLM: category=%s, year=%s",
                meta.doc_category,
                meta.academic_year,
            )
            return meta
    except Exception as e:
        logger.warning(
            "LLM meta failed for '%s': %s. Using regex fallback.", filename, e
        )
        _meta_cache[filename] = fallback
        return fallback
