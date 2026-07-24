"""
backend/services/registry.py
------------------------------
Pinecone has no native "list distinct metadata values" call, so there is
no way to ask it "which document names exist in namespace X". Previously
(single-process Streamlit app) that list lived in st.session_state. Now
that the frontend is a separate HTTP client talking to this API, that
bookkeeping has to live on the backend instead — this module is a small,
dependency-free JSON-backed registry for exactly that.

Not a database: fine for a student/demo deployment with light concurrent
usage. A production system would replace this with a real table (e.g. one
row per document in Postgres/SQLite) keyed by namespace.
"""

import json
import threading
from pathlib import Path

from backend.config import settings

_lock = threading.Lock()


def _read() -> dict[str, list[str]]:
    if not settings.registry_path.exists():
        return {}
    try:
        with open(settings.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write(data: dict[str, list[str]]) -> None:
    settings.registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.registry_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(settings.registry_path)  # atomic-ish swap


def add_document(namespace: str, document_name: str) -> None:
    with _lock:
        data = _read()
        docs = data.setdefault(namespace, [])
        if document_name not in docs:
            docs.append(document_name)
        _write(data)


def list_documents(namespace: str) -> list[str]:
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
