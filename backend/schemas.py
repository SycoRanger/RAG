"""
backend/schemas.py
--------------------
Pydantic models for every request/response body the API exposes. Keeping
these separate from backend/main.py means the route functions stay thin
and the data contracts are easy to find in one place.
"""

from pydantic import BaseModel, Field

from backend.config import settings


class UploadResponse(BaseModel):
    document_name: str
    namespace: str
    pages_extracted: int
    chunks_indexed: int
    ocr_mode: str
    pages_via_ocr: int  # how many pages needed OCR to be readable


class QueryRequest(BaseModel):
    question: str
    namespace: str
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(
        default_factory=lambda: settings.default_similarity_threshold, ge=0.0, le=1.0
    )
    document_filter: list[str] | None = None
    page_min: int | None = Field(default=None, ge=1)
    page_max: int | None = Field(default=None, ge=1)


class SourceItem(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    excerpt: str
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    used_context: bool
    confidence: float
    sources: list[SourceItem]


class DocumentInfo(BaseModel):
    name: str
    pages: int
    chunks: int
    ocr_pages: int
    uploaded_at: str


class DocumentsResponse(BaseModel):
    namespace: str
    documents: list[DocumentInfo]


class NamespacesResponse(BaseModel):
    namespaces: list[str]


class HealthResponse(BaseModel):
    status: str
    pinecone_connected: bool
    groq_configured: bool
    embedding_model: str
    embedding_device: str
    ocr_available: bool
    ocr_status: str


class LogEntry(BaseModel):
    timestamp_utc: str
    namespace: str
    question: str
    num_chunks_used: str
    used_context: str
    confidence: str
    answer_preview: str


class ErrorResponse(BaseModel):
    detail: str
