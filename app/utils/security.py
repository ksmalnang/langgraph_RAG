"""Security/privacy helpers for API policies and logging."""

from __future__ import annotations

import hashlib


def mask_session_id(session_id: str | None) -> str:
    """Return a short non-reversible identifier for logs."""
    if not session_id:
        return "none"
    digest = hashlib.sha256(session_id.encode()).hexdigest()
    return digest[:10]


def mask_email(email: str | None) -> str:
    """Mask email local-part while preserving domain for diagnostics."""
    if not email or "@" not in email:
        return "unknown"
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}"
    return f"{masked_local}@{domain}"


def parse_cors_origins(raw: str) -> list[str]:
    """Parse comma-separated CORS origins from config."""
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def is_local_env(app_env: str) -> bool:
    """Whether environment is local/dev style."""
    return app_env in {"development", "dev", "local", "test"}

