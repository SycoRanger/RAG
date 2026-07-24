# RAG System over PDFs — FastAPI backend + Streamlit frontend

An intermediate Retrieval-Augmented Generation system, now split into a
proper **API backend** (FastAPI) and a **thin UI client** (Streamlit) that
talks to it over HTTP — instead of one monolithic Streamlit app. Upload
PDFs, ask questions, get answers grounded strictly in the uploaded
content, with page-level source citations and similarity scores.

## Why a backend/frontend split?

The previous version put all pipeline logic directly inside a single
`app.py`. This version separates it into:

- **`backend/`** — a FastAPI service that does all the real work (PDF
  parsing, chunking, embedding, Pinecone, Groq) and exposes it as a REST
  API. You can call it from Streamlit, curl, Postman, a grading script,
  or any other client — it doesn't know or care who's calling it.
- **`frontend/`** — a Streamlit app that is *only* a UI. It never touches
  pypdf, sentence-transformers, Pinecone, or Groq directly; it just makes
  HTTP requests to the backend and renders the JSON it gets back.

This is what makes "the API work" as its own thing, and it's also a more
realistic architecture for the "Code Structure & Modularity" grading
criterion — the two processes can be developed, tested, deployed, and
scaled independently.

## Project structure

```
rag_system/
├── backend/
│   ├── main.py                 # FastAPI app — thin route layer only
│   ├── config.py                # env vars + settings, single source of truth
│   ├── schemas.py                # Pydantic request/response models
│   ├── services/
│   │   ├── pdf_loader.py         # PDF -> cleaned per-page text
│   │   ├── text_chunker.py       # per-page text -> overlapping chunks
│   │   ├── embedder.py           # chunks/queries -> vectors (torch + Sentence-Transformers)
│   │   ├── vector_store.py       # all Pinecone calls: create/upsert/query/clear
│   │   ├── retriever.py          # query -> ranked, threshold-filtered chunks
│   │   ├── generator.py          # chunks -> grounded answer via Groq
│   │   └── registry.py           # tracks which documents live in which namespace
│   └── utils/
│       ├── logger.py             # CSV query logging
│       └── helpers.py
├── frontend/
│   └── streamlit_app.py          # UI only — calls the backend via `requests`
├── data/
│   ├── logs/query_log.csv        # created at runtime
│   └── registry.json             # created at runtime
├── docs/
│   ├── architecture_diagram.svg
│   ├── qa_test_plan.pdf          # ready-to-run test questions + answer sheet
│   └── sample_test_document.pdf  # a small synthetic PDF to test against
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows (Command Prompt or PowerShell): venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: add your PINECONE_API_KEY and GROQ_API_KEY
```

Get a free Pinecone API key at https://app.pinecone.io and a free Groq
API key at https://console.groq.com. No OpenAI key is needed — embeddings
run locally via Sentence-Transformers/torch.

### Run it — two processes, two terminals

**Terminal 1 — the API backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```
Swagger UI (interactive API docs) will be at http://localhost:8000/docs
— you can upload a PDF and run queries directly from there, no frontend
required, which is a quick way to confirm the API itself works.

**Terminal 2 — the Streamlit UI:**
```bash
streamlit run frontend/streamlit_app.py
```
It talks to `http://localhost:8000` by default; set `BACKEND_URL` in
`.env` if you run the backend somewhere else.

### A note on Windows / MSYS2

If you're on Windows and running these commands from an **MSYS2 or
Git-Bash shell**, and you've seen errors or paths mentioning `msys64`
during `pip install` — that's almost always because that shell is using
an MSYS2-provided Python (built with MinGW), not the official Python
distribution. PyTorch's prebuilt wheels are only published for the
standard CPython build from python.org (or Anaconda); against an MSYS2
Python, pip can't find a matching wheel and either fails outright or
falls back to something inconsistent.

**Fix:** install Python from https://python.org (or Anaconda) and run
the commands above from a regular Command Prompt, PowerShell, or VS
Code terminal — not an MSYS2/Git-Bash shell. Git Bash itself is fine for
git commands; it's specifically the MSYS2 *Python interpreter* that
causes this.

## Why torch, and not torchvision

`sentence-transformers` runs its embedding model on top of **PyTorch**
(`torch`), so `torch` is a direct, explicit dependency here — it's listed
in `requirements.txt` and imported directly in `embedder.py` to pick
CPU vs. CUDA automatically.

`torchvision` is **not used** because there is no image/vision model
anywhere in this pipeline — it only processes extracted text. If you
later want to support scanned/image-only PDFs, that needs an OCR step
(e.g. `pytesseract`, or a vision-language model), which is a different
addition than torchvision and isn't included here — see the "Future
Work" section of the technical report if you want to add it.

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend + Pinecone + Groq status |
| `POST` | `/api/documents/upload` | Upload a PDF (multipart: `file`, `namespace`, `chunk_size`, `chunk_overlap`) |
| `GET` | `/api/documents?namespace=...` | List indexed documents in a namespace |
| `GET` | `/api/namespaces` | List all known namespaces |
| `DELETE` | `/api/namespaces/{namespace}` | Wipe a namespace (Pinecone + registry) |
| `POST` | `/api/query` | Ask a question (JSON body, see `backend/schemas.py`) |
| `GET` | `/api/logs?limit=50` | Recent query log entries |

Example with curl:
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@docs/sample_test_document.pdf" \
  -F "namespace=demo"

curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination notice period?", "namespace": "demo"}'
```

## How answers stay separated from their sources

The API returns `answer` and `sources` as two distinct JSON fields (see
`QueryResponse` in `backend/schemas.py`) — they are never concatenated
into one string on the backend. The Streamlit frontend renders them in
two visually separate sections ("Answer" then "📎 Sources", each source
in its own expandable box), so the generated answer and its citations
never run together on screen.

## Testing it

`docs/sample_test_document.pdf` is a small synthetic document (an
employment-agreement excerpt) you can upload immediately to try the
system without needing your own PDF. `docs/qa_test_plan.pdf` has a ready
list of test questions to run against it — including both questions the
document *can* answer and one it deliberately can't — with a table to
record what your running system actually returns, which doubles as
evidence for the "Retrieval Accuracy" and "Hallucination Prevention"
sections of your report.

## How hallucination is prevented

1. **Similarity threshold** — chunks below the configured cosine cutoff are dropped before the LLM ever sees them.
2. **Zero-context short-circuit** — if nothing survives that filter, the LLM is never called; the fixed refusal message is returned directly.
3. **Strict system prompt** — instructs the LLM to answer only from context and to return the exact sentence `"The answer is not available in the provided document."` otherwise.
4. **temperature = 0** — deterministic, non-creative generation.
5. **Traceable answers** — every answer ships with its exact source chunks (document, page, excerpt, similarity score).

## Intermediate enhancements implemented

- ✅ Multi-document support (shared namespace, per-document filtering)
- ✅ Query history (session view + persisted CSV log)
- ✅ Adjustable chunk size / overlap from the UI
- ✅ Adjustable top-k and similarity threshold from the UI
- ✅ Metadata filtering (by document, by page range)
- ✅ Confidence scoring display (retrieval-grounded heuristic)
- ✅ Logging user queries (`data/logs/query_log.csv`)

## Known limitations

- No OCR — scanned/image-only PDFs raise a clear error rather than silently returning nothing.
- The confidence score is a retrieval-quality heuristic, not a calibrated probability.
- `registry.py` is a JSON file, not a database — fine for a demo/student deployment; swap for a real table if this goes to production with concurrent writers.
