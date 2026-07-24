"""
backend/utils/helpers.py
--------------------------
Small, generic helpers that don't belong to any single service.
"""

import re


def safe_namespace(name: str) -> str:
    """Turn an arbitrary string into a Pinecone-safe namespace."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", name.strip())
    return cleaned[:100] or "default"


def bytes_to_mb(n_bytes: int) -> float:
    return round(n_bytes / (1024 * 1024), 2)


def truncate(text: str, max_chars: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "…"
