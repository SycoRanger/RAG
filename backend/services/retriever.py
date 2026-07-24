"""
backend/services/retriever.py
-------------------------------
Turns a user question into ranked, threshold-filtered context chunks.
Sits between the embedder + vector_store and the answer generator.
"""

from dataclasses import dataclass

from backend.services import vector_store
from backend.services.embedder import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str
    score: float  # cosine similarity, 0-1 (higher = more relevant)


def retrieve(
    question: str,
    namespace: str,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
    document_filter: list[str] | None = None,
    page_filter: tuple[int, int] | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for `question`, dropping any
    below `similarity_threshold`. An empty result means "not enough
    relevant context" and the generator must refuse to answer.
    """
    if not question or not question.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = embed_query(question.strip())
    raw_matches = vector_store.query(
        query_vector=query_vector,
        namespace=namespace,
        top_k=top_k,
        document_filter=document_filter,
        page_filter=page_filter,
    )

    results = [
        RetrievedChunk(
            chunk_id=m["chunk_id"],
            document_name=m["document_name"],
            page_number=m["page_number"],
            text=m["text"],
            score=m["score"],
        )
        for m in raw_matches
        if m["score"] >= similarity_threshold
    ]
    return results
