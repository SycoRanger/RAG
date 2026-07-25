"""
backend/main.py
------------------
The FastAPI application. Every /api/* route here is thin: parse input,
call a service function, translate its result/exception into an HTTP
response. No PDF parsing, chunking, embedding, Pinecone, or LLM logic
lives in this file — that all stays in backend/services/.

This file ALSO serves the HTML GUI (frontend/static/index.html) at "/",
so the whole app is one process on one port: no separate frontend server,
no CORS to configure for normal use.

Run with:
    uvicorn backend.main:app --reload --port 8000

Then open http://localhost:8000 in a browser for the GUI, or
http://localhost:8000/docs for the interactive API docs (Swagger UI).
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.schemas import (
    DocumentInfo, DocumentsResponse, ErrorResponse, HealthResponse, LogEntry,
    NamespacesResponse, QueryRequest, QueryResponse, SourceItem, UploadResponse,
)
from backend.services import registry, vector_store
from backend.services.embedder import device_in_use, embed_texts
from backend.services.generator import generate_answer
from backend.services.pdf_loader import InvalidPDFError, load_pdf, ocr_status
from backend.services.retriever import retrieve
from backend.services.text_chunker import chunk_pages
from backend.services.vector_store import VectorStoreError
from backend.utils.helpers import safe_namespace, truncate
from backend.utils.logger import log_query, read_recent_logs

app = FastAPI(
    title="RAG over PDFs — Pinecone + Groq",
    description="Upload PDFs, index them in Pinecone, and ask questions answered strictly from their content.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health():
    ocr_ok, ocr_msg = ocr_status()
    return HealthResponse(
        status="ok",
        pinecone_connected=vector_store.check_connection(),
        groq_configured=bool(settings.groq_api_key),
        embedding_model=settings.embedding_model_name,
        embedding_device=device_in_use(),
        ocr_available=ocr_ok,
        ocr_status=ocr_msg,
    )


@app.post("/api/documents/upload", response_model=UploadResponse, tags=["documents"],
          responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
async def upload_document(
    file: UploadFile = File(...),
    namespace: str = Form(default=None),
    chunk_size: int = Form(default=settings.default_chunk_size),
    chunk_overlap: int = Form(default=settings.default_chunk_overlap),
    ocr_mode: str = Form(default=settings.default_ocr_mode),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"'{file.filename}' is not a PDF file.")

    namespace = safe_namespace(namespace) if namespace else f"session-{uuid.uuid4().hex[:8]}"
    file_bytes = await file.read()

    try:
        pages = load_pdf(file_bytes, file.filename, max_mb=settings.max_upload_mb, ocr_mode=ocr_mode)
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except InvalidPDFError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=400, detail=f"'{file.filename}' produced no usable text chunks.")

    try:
        vectors = embed_texts([c.text for c in chunks])
        indexed = vector_store.upsert_chunks(chunks, vectors, namespace=namespace)
    except VectorStoreError as e:
        raise HTTPException(status_code=502, detail=str(e))

    registry.add_document(
        namespace,
        file.filename,
        pages=len(pages),
        chunk_ids=[c.chunk_id for c in chunks],
        ocr_pages=sum(1 for p in pages if p.via_ocr),
    )

    return UploadResponse(
        document_name=file.filename,
        namespace=namespace,
        pages_extracted=len(pages),
        chunks_indexed=indexed,
        ocr_mode=ocr_mode,
        pages_via_ocr=sum(1 for p in pages if p.via_ocr),
    )


@app.get("/api/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents(namespace: str = Query(...)):
    records = registry.list_documents(namespace)
    return DocumentsResponse(
        namespace=namespace,
        documents=[
            DocumentInfo(
                name=d["name"],
                pages=d.get("pages", 0),
                chunks=len(d.get("chunk_ids", [])),
                ocr_pages=d.get("ocr_pages", 0),
                uploaded_at=d.get("uploaded_at", ""),
            )
            for d in records
        ],
    )


@app.delete("/api/documents/{document_name}", tags=["documents"],
            responses={404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
def delete_document(document_name: str, namespace: str = Query(...)):
    """Delete a single document: its vectors (by exact chunk ID, not
    metadata filter -- see vector_store.delete_by_ids for why) and its
    registry record."""
    record = registry.get_document(namespace, document_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"'{document_name}' not found in namespace '{namespace}'.")
    try:
        vector_store.delete_by_ids(record.get("chunk_ids", []), namespace=namespace)
    except VectorStoreError as e:
        raise HTTPException(status_code=502, detail=str(e))
    registry.remove_document(namespace, document_name)
    return {"status": "deleted", "namespace": namespace, "document_name": document_name}


@app.get("/api/namespaces", response_model=NamespacesResponse, tags=["documents"])
def list_namespaces():
    return NamespacesResponse(namespaces=registry.list_namespaces())


@app.delete("/api/namespaces/{namespace}", tags=["documents"],
            responses={502: {"model": ErrorResponse}})
def delete_namespace(namespace: str):
    try:
        vector_store.clear_namespace(namespace)
    except VectorStoreError as e:
        raise HTTPException(status_code=502, detail=str(e))
    registry.clear_namespace(namespace)
    return {"status": "cleared", "namespace": namespace}


@app.post("/api/query", response_model=QueryResponse, tags=["query"],
          responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
def query(req: QueryRequest):
    page_filter = None
    if req.page_min is not None and req.page_max is not None:
        if req.page_min > req.page_max:
            raise HTTPException(status_code=400, detail="page_min cannot be greater than page_max.")
        page_filter = (req.page_min, req.page_max)

    try:
        chunks = retrieve(
            question=req.question,
            namespace=req.namespace,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
            document_filter=req.document_filter,
            page_filter=page_filter,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except VectorStoreError as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        result = generate_answer(req.question, chunks)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    log_query(
        namespace=req.namespace,
        question=req.question,
        num_chunks_used=len(result.sources),
        used_context=result.used_context,
        confidence=result.confidence,
        answer=result.answer,
    )

    return QueryResponse(
        answer=result.answer,
        used_context=result.used_context,
        confidence=result.confidence,
        sources=[
            SourceItem(
                document_name=s.document_name,
                page_number=s.page_number,
                chunk_id=s.chunk_id,
                excerpt=truncate(s.text, 500),
                similarity_score=s.score,
            )
            for s in result.sources
        ],
    )


@app.get("/api/logs", response_model=list[LogEntry], tags=["system"])
def logs(limit: int = Query(default=50, ge=1, le=500)):
    return read_recent_logs(limit=limit)


# ---------------------------------------------------------------------------
# Serve the HTML GUI at "/". This MUST come after every @app.get/post/delete
# route above, since StaticFiles("/", html=True) is a catch-all — Starlette
# matches routes in registration order, so the specific /api/* routes above
# are checked first and this only picks up whatever they didn't handle.
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="gui")
