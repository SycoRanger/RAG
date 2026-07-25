"""
backend/services/vector_store.py
----------------------------------
All direct interaction with Pinecone lives here: connecting, creating the
index if it doesn't exist, upserting vectors with metadata, namespaced
querying, and housekeeping. Every other module talks to Pinecone only
through this file.

Metadata stored per vector:
    - document_name : original PDF filename
    - page_number   : 1-indexed page the chunk came from
    - chunk_id       : stable id, also used as the vector id
    - text          : the chunk text itself
"""

import time

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException

from backend.config import settings
from backend.services.text_chunker import Chunk

_UPSERT_BATCH_SIZE = 100


class VectorStoreError(Exception):
    """Raised for any Pinecone connection / operation failure."""


def _client() -> Pinecone:
    try:
        return Pinecone(api_key=settings.pinecone_api_key)
    except PineconeException as e:
        raise VectorStoreError(f"Could not connect to Pinecone: {e}") from e


def ensure_index(dimension: int = settings.embedding_dimension):
    """Create the configured index if it doesn't already exist, then return a handle to it."""
    pc = _client()
    try:
        existing = {idx["name"] for idx in pc.list_indexes()}
        if settings.pinecone_index_name not in existing:
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=dimension,
                metric=settings.pinecone_metric,
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
            while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
                time.sleep(1)
        return pc.Index(settings.pinecone_index_name)
    except PineconeException as e:
        raise VectorStoreError(f"Pinecone index setup failed: {e}") from e


def check_connection() -> bool:
    """Lightweight, side-effect-free health check used by GET /api/health.
    Deliberately does NOT call ensure_index(), since that can create an
    index as a side effect — a health check should only ever read."""
    try:
        pc = _client()
        pc.list_indexes()
        return True
    except PineconeException:
        return False


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]], namespace: str) -> int:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length.")
    if not chunks:
        return 0

    index = ensure_index(dimension=len(vectors[0]))
    records = [
        {
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": {
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    try:
        total = 0
        for i in range(0, len(records), _UPSERT_BATCH_SIZE):
            batch = records[i : i + _UPSERT_BATCH_SIZE]
            index.upsert(vectors=batch, namespace=namespace)
            total += len(batch)
        return total
    except PineconeException as e:
        raise VectorStoreError(f"Upsert to Pinecone failed: {e}") from e


def query(
    query_vector: list[float],
    namespace: str,
    top_k: int = 5,
    document_filter: list[str] | None = None,
    page_filter: tuple[int, int] | None = None,
) -> list[dict]:
    index = ensure_index(dimension=len(query_vector))

    pinecone_filter: dict = {}
    if document_filter:
        pinecone_filter["document_name"] = {"$in": document_filter}
    if page_filter:
        lo, hi = page_filter
        pinecone_filter["page_number"] = {"$gte": lo, "$lte": hi}

    try:
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=pinecone_filter or None,
        )
    except PineconeException as e:
        raise VectorStoreError(f"Query to Pinecone failed: {e}") from e

    matches = []
    for m in result.get("matches", []):
        meta = m.get("metadata", {})
        matches.append(
            {
                "chunk_id": meta.get("chunk_id"),
                "document_name": meta.get("document_name"),
                "page_number": meta.get("page_number"),
                "text": meta.get("text", ""),
                "score": float(m.get("score", 0.0)),
            }
        )
    return matches


def clear_namespace(namespace: str) -> None:
    index = ensure_index()
    try:
        index.delete(delete_all=True, namespace=namespace)
    except PineconeException as e:
        raise VectorStoreError(f"Failed to clear namespace '{namespace}': {e}") from e


def delete_by_ids(ids: list[str], namespace: str) -> None:
    """Delete specific vectors by their exact IDs. Used for single-document
    removal -- explicit-ID delete is universally supported on Pinecone
    serverless, unlike metadata-filter delete, whose support has varied
    across serverless index versions."""
    if not ids:
        return
    index = ensure_index()
    try:
        # Pinecone caps delete batch size; chunk defensively same as upsert.
        for i in range(0, len(ids), _UPSERT_BATCH_SIZE):
            index.delete(ids=ids[i : i + _UPSERT_BATCH_SIZE], namespace=namespace)
    except PineconeException as e:
        raise VectorStoreError(f"Failed to delete document vectors: {e}") from e
