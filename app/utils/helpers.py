"""Miscellaneous helper functions."""

from __future__ import annotations

import hashlib
import uuid


def generate_doc_id(filename: str) -> str:
    """Deterministic document ID from filename."""
    return hashlib.sha256(filename.encode()).hexdigest()[:16]


def generate_point_id() -> str:
    """Random UUID string for Qdrant point IDs."""
    return str(uuid.uuid4())


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text with an ellipsis if it exceeds *max_chars*."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
