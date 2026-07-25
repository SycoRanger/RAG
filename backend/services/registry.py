"""
backend/services/registry.py
------------------------------
Pinecone has no native "list distinct metadata values" call, so there is
no way to ask it "which document names exist in namespace X", let alone
richer per-document facts like page/chunk counts. Previously (single-
process Streamlit app) that lived in st.session_state. Now that the
frontend is a separate HTTP client, that bookkeeping lives here instead --
a small, dependency-free JSON-backed registry.

Each namespace maps to a list of document records (not just names), so a
"document library" UI can show real page counts, chunk counts, OCR usage,
and upload time -- all facts this app actually knows, not placeholders.

Not a database: fine for a student/demo deployment with light concurrent
usage. A production system would replace this with a real table (e.g. one
row per document in Postgres/SQLite) keyed by namespace.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.config import settings

_lock = threading.Lock()


def _read() -> dict[str, list[dict]]:
    if not settings.registry_path.exists():
        return {}
    try:
        with open(settings.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write(data: dict[str, list[dict]]) -> None:
    settings.registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.registry_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(settings.registry_path)  # atomic-ish swap


def add_document(
    namespace: str,
    document_name: str,
    *,
    pages: int = 0,
    chunk_ids: list[str] | None = None,
    ocr_pages: int = 0,
) -> None:
    """Record (or re-record, on re-upload) a document's real stats.

    chunk_ids is stored in full (not just a count) so a single document
    can later be deleted precisely via Pinecone's explicit-ID delete --
    metadata-filter delete has inconsistent support across serverless
    index versions, but delete-by-ID-list is universally supported.
    """
    with _lock:
        data = _read()
        docs = data.setdefault(namespace, [])
        docs = [d for d in docs if d.get("name") != document_name]
        docs.append({
            "name": document_name,
            "pages": pages,
            "chunk_ids": chunk_ids or [],
            "ocr_pages": ocr_pages,
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        data[namespace] = docs
        _write(data)


def list_documents(namespace: str) -> list[dict]:
    with _lock:
        return list(_read().get(namespace, []))


def list_namespaces() -> list[str]:
    with _lock:
        return list(_read().keys())


def clear_namespace(namespace: str) -> None:
    with _lock:
        data = _read()
        data.pop(namespace, None)
        _write(data)


def get_document(namespace: str, document_name: str) -> dict | None:
    with _lock:
        for d in _read().get(namespace, []):
            if d.get("name") == document_name:
                return d
        return None


def remove_document(namespace: str, document_name: str) -> None:
    """Drop one document's record from a namespace's registry entry.
    Callers are responsible for also deleting its vectors from Pinecone
    (see vector_store.delete_document) -- this only updates bookkeeping."""
    with _lock:
        data = _read()
        docs = data.get(namespace, [])
        data[namespace] = [d for d in docs if d.get("name") != document_name]
        _write(data)
