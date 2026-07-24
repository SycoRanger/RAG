"""
frontend/streamlit_app.py
----------------------------
The interface. Every action here is an HTTP call to the FastAPI backend —
this file imports no pypdf, torch, pinecone, or groq.

Design intent: the job of this screen is to make it obvious WHERE an answer
came from and WHEN the system doesn't know. So retrieved passages are
presented as numbered exhibits with a page stamp and a similarity meter,
and the "not in the document" state is styled as a neutral finding (ochre)
rather than an error (red) — a refusal is the system working correctly,
and colouring it red would teach you to read correct behaviour as failure.

Run with (backend must already be running):
    streamlit run frontend/streamlit_app.py
"""

import os
import uuid
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Evidence — RAG over PDFs",
    page_icon="§",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- design tokens
CSS = """
<style>
:root {
  --ink:      #16202B;
  --paper:    #F7F8FA;
  --rule:     #DDE3EA;
  --seal:     #2F5D8C;
  --verify:   #1F7A5C;
  --absent:   #8A6D3B;
  --muted:    #64748B;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
}

/* ---- masthead ---- */
.rag-masthead {
  border-bottom: 2px solid var(--ink);
  padding-bottom: 10px; margin-bottom: 4px;
}
.rag-masthead h1 {
  font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--ink); margin: 0 0 2px 0;
}
.rag-masthead .sub {
  font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted);
}

/* ---- status pills ---- */
.rag-pills { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px 0; }
.rag-pill {
  font-family: var(--mono); font-size: 0.69rem; letter-spacing: 0.04em;
  padding: 3px 9px; border-radius: 2px; border: 1px solid var(--rule);
  background: #fff; color: var(--muted); white-space: nowrap;
}
.rag-pill.on  { border-color: var(--verify); color: var(--verify); }
.rag-pill.off { border-color: var(--absent); color: var(--absent); }

/* ---- answer block ---- */
.rag-answer {
  border-left: 3px solid var(--verify); background: #fff;
  padding: 16px 18px; margin: 6px 0 4px 0; border-radius: 0 3px 3px 0;
  border-top: 1px solid var(--rule); border-right: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
}
.rag-answer.absent { border-left-color: var(--absent); }
.rag-answer .eyebrow {
  font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted); display: block; margin-bottom: 7px;
}
.rag-answer .body { font-size: 1.02rem; line-height: 1.6; color: var(--ink); }

/* ---- exhibit (source) cards ---- */
.rag-exhibit {
  border: 1px solid var(--rule); background: #fff; border-radius: 3px;
  padding: 12px 14px; margin-bottom: 8px;
}
.rag-exhibit .head {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 12px; margin-bottom: 8px;
}
.rag-exhibit .stamp {
  font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.06em;
  color: var(--seal); border: 1px solid var(--seal);
  padding: 2px 7px; border-radius: 2px; white-space: nowrap;
}
.rag-exhibit .docname {
  font-family: var(--mono); font-size: 0.72rem; color: var(--muted);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.rag-exhibit .excerpt {
  font-size: 0.88rem; line-height: 1.55; color: var(--ink);
  border-left: 2px solid var(--rule); padding-left: 10px;
}
/* similarity meter */
.rag-meter { margin-top: 9px; }
.rag-meter .track {
  height: 3px; background: var(--rule); border-radius: 2px; overflow: hidden;
}
.rag-meter .fill { height: 3px; background: var(--seal); }
.rag-meter .label {
  font-family: var(--mono); font-size: 0.66rem; color: var(--muted);
  margin-top: 4px; display: block;
}

/* ---- key/value rows for the system tab ---- */
.rag-kv {
  display: flex; justify-content: space-between; gap: 14px;
  border-bottom: 1px dotted var(--rule); padding: 6px 0;
  font-size: 0.84rem;
}
.rag-kv .k { color: var(--muted); }
.rag-kv .v { font-family: var(--mono); color: var(--ink); text-align: right; }

/* ---- empty state ---- */
.rag-empty {
  border: 1px dashed var(--rule); border-radius: 3px; padding: 22px;
  text-align: center; color: var(--muted); font-size: 0.9rem; background: #fff;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- api helpers
def api_get(path, **kw):
    return requests.get(f"{BACKEND_URL}{path}", timeout=30, **kw)


def api_post(path, **kw):
    return requests.post(f"{BACKEND_URL}{path}", timeout=600, **kw)


def api_delete(path, **kw):
    return requests.delete(f"{BACKEND_URL}{path}", timeout=30, **kw)


def error_detail(resp) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"


def esc(text: str) -> str:
    """Escape user/document text before putting it inside our HTML cards."""
    return (
        str(text)
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------- connectivity
try:
    health = api_get("/api/health").json()
except requests.exceptions.ConnectionError:
    st.markdown(
        '<div class="rag-masthead"><h1>Document Evidence</h1>'
        '<span class="sub">backend unreachable</span></div>',
        unsafe_allow_html=True,
    )
    st.error(
        f"No backend at {BACKEND_URL}. Start it in a separate terminal, from the "
        f"project root:\n\n`uvicorn backend.main:app --reload --port 8000`"
    )
    st.stop()

# ---------------------------------------------------------------- session state
if "namespace" not in st.session_state:
    st.session_state.namespace = f"session-{uuid.uuid4().hex[:8]}"
if "history" not in st.session_state:
    st.session_state.history = []
if "upload_results" not in st.session_state:
    st.session_state.upload_results = []

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("#### Knowledge base")
    st.text_input(
        "Namespace",
        key="namespace",
        help="Documents indexed under the same namespace are searched together.",
    )

    st.markdown("#### Reading")
    ocr_choice = st.radio(
        "OCR for scanned pages",
        options=["auto", "off", "force"],
        index=0,
        format_func=lambda v: {
            "auto": "Auto — only pages with no text",
            "off": "Off — skip image pages",
            "force": "Always — re-read every page",
        }[v],
        help=(
            "Auto is right for almost everything: typed pages are read directly "
            "and only image pages go through OCR. Use Always when a PDF has a "
            "text layer that came out garbled."
        ),
    )
    if not health.get("ocr_available") and ocr_choice != "off":
        st.warning(f"OCR unavailable — {health.get('ocr_status', 'unknown')}", icon="⚠")

    st.markdown("#### Chunking")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 120, step=20)

    st.markdown("#### Retrieval")
    top_k = st.slider("Passages to retrieve", 1, 15, 5)
    similarity_threshold = st.slider(
        "Similarity floor", 0.0, 1.0, 0.25, step=0.05,
        help="Passages scoring below this never reach the language model. "
             "Raise it to make the system refuse more readily.",
    )

    docs_resp = api_get("/api/documents", params={"namespace": st.session_state.namespace})
    documents = docs_resp.json().get("documents", []) if docs_resp.ok else []

    doc_filter, page_range = [], None
    if documents:
        st.markdown("#### Narrow the search")
        doc_filter = st.multiselect("Only these documents", options=documents)
        if st.checkbox("Only a page range"):
            page_range = st.slider("Pages", 1, 500, (1, 50))

    st.divider()
    with st.expander("Clear this knowledge base"):
        st.caption("Deletes every indexed passage in this namespace. Can't be undone.")
        if st.button("Delete all passages", use_container_width=True):
            resp = api_delete(f"/api/namespaces/{st.session_state.namespace}")
            if resp.ok:
                st.session_state.upload_results = []
                st.success("Cleared.")
                st.rerun()
            else:
                st.error(error_detail(resp))

# ---------------------------------------------------------------- masthead
st.markdown(
    '<div class="rag-masthead">'
    "<h1>Document Evidence</h1>"
    '<span class="sub">answers traced to the page they came from</span>'
    "</div>",
    unsafe_allow_html=True,
)

pills = [
    ("Pinecone", health["pinecone_connected"]),
    ("Groq", health["groq_configured"]),
    ("OCR", health.get("ocr_available", False)),
]
pill_html = "".join(
    f'<span class="rag-pill {"on" if ok else "off"}">{name} {"ready" if ok else "unavailable"}</span>'
    for name, ok in pills
)
pill_html += f'<span class="rag-pill">{esc(health["embedding_model"])} · {esc(health["embedding_device"])}</span>'
pill_html += f'<span class="rag-pill">ns: {esc(st.session_state.namespace)}</span>'
st.markdown(f'<div class="rag-pills">{pill_html}</div>', unsafe_allow_html=True)

tab_ask, tab_add, tab_history, tab_system = st.tabs(
    ["Ask", "Add documents", "History", "System"]
)

# ---------------------------------------------------------------- TAB: add
with tab_add:
    st.markdown("##### Add PDFs to this knowledge base")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.caption(
        f"Up to 20 MB each. Scanned and photographed pages are handled by OCR "
        f"(currently: **{ocr_choice}**)."
    )

    if uploaded_files and st.button("Read and index", type="primary"):
        progress = st.progress(0.0, text="Starting")
        results = []
        for i, uf in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Reading {uf.name}")
            resp = api_post(
                "/api/documents/upload",
                files={"file": (uf.name, uf.getvalue(), "application/pdf")},
                data={
                    "namespace": st.session_state.namespace,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "ocr_mode": ocr_choice,
                },
            )
            results.append(
                {"name": uf.name, "ok": resp.ok,
                 "body": resp.json() if resp.ok else error_detail(resp)}
            )
        progress.progress(1.0, text="Done")
        st.session_state.upload_results = results
        st.rerun()

    if st.session_state.upload_results:
        st.markdown("##### Last run")
        for r in st.session_state.upload_results:
            if r["ok"]:
                b = r["body"]
                ocr_line = (
                    f" · {b['pages_via_ocr']} page(s) recovered by OCR"
                    if b.get("pages_via_ocr") else ""
                )
                st.success(
                    f"**{r['name']}** — {b['pages_extracted']} pages, "
                    f"{b['chunks_indexed']} passages indexed{ocr_line}"
                )
            else:
                st.error(f"**{r['name']}** — {r['body']}")

    st.markdown("##### In this knowledge base")
    if documents:
        for d in documents:
            st.markdown(f"- {d}")
    else:
        st.markdown(
            '<div class="rag-empty">Nothing indexed yet. Add a PDF above to start asking questions.</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- TAB: ask
with tab_ask:
    if not documents:
        st.markdown(
            '<div class="rag-empty">No documents in this knowledge base yet.<br>'
            "Open <b>Add documents</b> to index a PDF first.</div>",
            unsafe_allow_html=True,
        )
    else:
        question = st.text_input(
            "Question",
            placeholder="What is the termination notice period?",
            label_visibility="collapsed",
        )
        ask = st.button("Ask", type="primary")

        if ask:
            if not question.strip():
                st.warning("Type a question first.")
            else:
                with st.spinner("Searching the documents"):
                    resp = api_post("/api/query", json={
                        "question": question,
                        "namespace": st.session_state.namespace,
                        "top_k": top_k,
                        "similarity_threshold": similarity_threshold,
                        "document_filter": doc_filter or None,
                        "page_min": page_range[0] if page_range else None,
                        "page_max": page_range[1] if page_range else None,
                    })

                if not resp.ok:
                    st.error(error_detail(resp))
                else:
                    result = resp.json()
                    grounded = result["used_context"]

                    st.markdown(
                        f'<div class="rag-answer {"" if grounded else "absent"}">'
                        f'<span class="eyebrow">'
                        f'{"Answer — drawn from the documents" if grounded else "Not found in these documents"}'
                        f"</span>"
                        f'<div class="body">{esc(result["answer"])}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if grounded:
                        conf = result["confidence"]
                        st.markdown(
                            f'<div class="rag-meter"><div class="track">'
                            f'<div class="fill" style="width:{conf*100:.0f}%"></div></div>'
                            f'<span class="label">retrieval confidence {conf:.0%} · '
                            f'{len(result["sources"])} passage(s) used</span></div>',
                            unsafe_allow_html=True,
                        )

                        st.markdown("##### Where this came from")
                        for n, s in enumerate(result["sources"], start=1):
                            score = s["similarity_score"]
                            st.markdown(
                                f'<div class="rag-exhibit">'
                                f'<div class="head">'
                                f'<span class="stamp">{n} · PAGE {s["page_number"]}</span>'
                                f'<span class="docname">{esc(s["document_name"])}</span>'
                                f"</div>"
                                f'<div class="excerpt">{esc(s["excerpt"])}</div>'
                                f'<div class="rag-meter"><div class="track">'
                                f'<div class="fill" style="width:{score*100:.0f}%"></div></div>'
                                f'<span class="label">similarity {score:.3f}</span></div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption(
                            "Nothing in the indexed pages scored above the similarity "
                            "floor. Lower it in the sidebar, or add the document that "
                            "covers this."
                        )

                    st.session_state.history.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "question": question,
                        "answer": result["answer"],
                        "grounded": grounded,
                        "confidence": result["confidence"],
                        "sources": [
                            f'{s["document_name"]} p.{s["page_number"]} ({s["similarity_score"]:.2f})'
                            for s in result["sources"]
                        ],
                    })

# ---------------------------------------------------------------- TAB: history
with tab_history:
    st.markdown("##### This session")
    if not st.session_state.history:
        st.markdown('<div class="rag-empty">No questions asked yet.</div>',
                    unsafe_allow_html=True)
    else:
        for item in st.session_state.history:
            st.markdown(
                f'<div class="rag-answer {"" if item["grounded"] else "absent"}">'
                f'<span class="eyebrow">{item["time"]} · {esc(item["question"])}</span>'
                f'<div class="body">{esc(item["answer"])}</div></div>',
                unsafe_allow_html=True,
            )
            if item["sources"]:
                st.caption(
                    " · ".join(item["sources"]) + f"  —  confidence {item['confidence']:.0%}"
                )

    st.markdown("##### Saved query log")
    st.caption("Written to data/logs/query_log.csv on the backend, kept across restarts.")
    logs_resp = api_get("/api/logs", params={"limit": 50})
    logs = logs_resp.json() if logs_resp.ok else []
    if logs:
        st.dataframe(logs, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="rag-empty">Log is empty.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- TAB: system
with tab_system:
    left, right = st.columns(2)

    with left:
        st.markdown("##### Services")
        rows = [
            ("Backend", BACKEND_URL),
            ("Pinecone", "connected" if health["pinecone_connected"] else "not connected"),
            ("Groq API key", "configured" if health["groq_configured"] else "missing"),
            ("Embedding model", health["embedding_model"]),
            ("Compute device", health["embedding_device"]),
            ("OCR", health.get("ocr_status", "unknown")),
        ]
        for k, v in rows:
            st.markdown(
                f'<div class="rag-kv"><span class="k">{k}</span>'
                f'<span class="v">{esc(v)}</span></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("##### Current settings")
        rows = [
            ("Namespace", st.session_state.namespace),
            ("OCR mode", ocr_choice),
            ("Chunk size", f"{chunk_size} chars"),
            ("Chunk overlap", f"{chunk_overlap} chars"),
            ("Passages retrieved", str(top_k)),
            ("Similarity floor", f"{similarity_threshold:.2f}"),
            ("Documents indexed", str(len(documents))),
        ]
        for k, v in rows:
            st.markdown(
                f'<div class="rag-kv"><span class="k">{k}</span>'
                f'<span class="v">{esc(v)}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("##### API")
    st.caption(
        f"The backend is a standalone REST service — you can drive it without this "
        f"interface. Interactive docs: {BACKEND_URL}/docs"
    )
    st.code(
        "POST   /api/documents/upload    file, namespace, chunk_size, chunk_overlap, ocr_mode\n"
        "GET    /api/documents           ?namespace=...\n"
        "GET    /api/namespaces\n"
        "DELETE /api/namespaces/{name}\n"
        "POST   /api/query               question, namespace, top_k, similarity_threshold\n"
        "GET    /api/logs                ?limit=50\n"
        "GET    /api/health",
        language="text",
    )

    if not health.get("ocr_available"):
        st.markdown("##### Enabling OCR")
        st.caption(
            "OCR needs the Tesseract program, which pip can't install because it "
            "isn't a Python package."
        )
        st.markdown(
            "1. Windows: install from the "
            "[UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki). "
            "macOS: `brew install tesseract`. Debian/Ubuntu: `sudo apt install tesseract-ocr`.\n"
            "2. On Windows, add this line to `.env` and restart the backend:\n"
        )
        st.code(r'TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe', language="text")
