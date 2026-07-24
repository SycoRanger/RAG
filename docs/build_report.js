const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, LevelFormat, convertInchesToTwip,
} = require("docx");

const PAGE_W = 12240, PAGE_H = 15840; // US Letter DXA
const bodyFont = "Calibri";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 140 },
    children: [new TextRun({ text, font: bodyFont, size: 22, ...opts })],
  });
}
function pMixed(runs) {
  return new Paragraph({ spacing: { after: 140 }, children: runs.map(r => new TextRun({ font: bodyFont, size: 22, ...r })) });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: bodyFont, size: 22 })],
  });
}
function cell(text, { bold = false, shade = null, width } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold, font: bodyFont, size: 20 })] })],
  });
}

const configRows = [
  ["Setting", "Value", "Notes"],
  ["Cloud provider", "aws", "Configurable via PINECONE_CLOUD"],
  ["Region", "us-east-1", "Configurable via PINECONE_REGION"],
  ["Index type", "Serverless", "Auto-created on first run if absent"],
  ["Metric", "cosine", "Matches normalized Sentence-Transformer embeddings"],
  ["Dimension", "384", "Matches all-MiniLM-L6-v2 output size"],
  ["Namespace scheme", "One namespace per knowledge base / session", "Enables multi-document isolation and one-click reset"],
  ["Metadata fields", "document_name, page_number, chunk_id, text", "Powers source attribution and metadata filtering"],
  ["Upsert batch size", "100 vectors/request", "Follows Pinecone's recommended batching practice"],
];
const colW = [3600, 3600, 2960];
function headerCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "4338CA" },
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", font: bodyFont, size: 20 })] })],
  });
}
function dataTable(headers, rows, widths) {
  const headerRow = new TableRow({ children: headers.map((h, j) => headerCell(h, widths[j])) });
  const bodyRows = rows.map((r, i) => new TableRow({
    children: r.map((t, j) => cell(t, { shade: i % 2 === 0 ? "F1F5F9" : null, width: widths[j] })),
  }));
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, rows: [headerRow, ...bodyRows] });
}

const diagramBuffer = fs.readFileSync(__dirname + "/architecture_diagram.png");

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.2) } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Intermediate RAG System Using Pinecone Vector Database", bold: true, size: 34, font: bodyFont })],
        spacing: { after: 80 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Technical Report", size: 26, color: "6B7280", font: bodyFont })],
        spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Stack: FastAPI backend \u00B7 Streamlit frontend \u00B7 Pinecone (serverless) \u00B7 Sentence-Transformers + torch (local embeddings) \u00B7 Groq LLM API", size: 20, italics: true, color: "6B7280", font: bodyFont })],
        spacing: { after: 300 },
      }),

      h1("1. Objective"),
      p("This project implements a Retrieval-Augmented Generation (RAG) system that answers questions strictly from the content of user-uploaded PDF documents. The system extracts and chunks PDF text, embeds it locally, indexes it in Pinecone, retrieves the most relevant passages for a given question, and generates an answer using an LLM that is explicitly restricted to the retrieved context. Every answer is paired with page-level source citations and a similarity score so it can be independently verified against the source document, and the system returns a fixed refusal message rather than a fabricated answer when the document does not contain enough information."),

      h1("2. System Architecture"),
      p("The pipeline follows the required eight-stage flow: PDF Upload \u2192 Text Extraction \u2192 Text Chunking \u2192 Embedding Generation \u2192 Pinecone Indexing \u2192 Semantic Retrieval \u2192 LLM Response Generation \u2192 Answer with Source Reference. Unlike a single monolithic script, this implementation splits the system into two independently runnable processes: a FastAPI backend (backend/) that owns all pipeline logic and exposes it as a REST API, and a Streamlit frontend (frontend/streamlit_app.py) that is a pure UI client calling that API over HTTP. The frontend contains no PDF-parsing, embedding, Pinecone, or LLM code at all \u2014 it only sends requests and renders JSON responses. This means the API can equally be driven from Swagger UI, curl, or an automated grading script, independent of any particular front end."),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 100 },
        children: [new ImageRun({ data: diagramBuffer, transformation: { width: 520, height: 360 }, type: "png" })],
      }),
      p("Figure 1: End-to-end pipeline and module mapping, including the two external services (Pinecone and Groq) and the one local component (the embedding model).", { italics: true, size: 18, color: "6B7280" }),

      h1("3. Design Decisions"),
      h2("3.1 Backend/frontend separation"),
      p("All pipeline logic lives under backend/services/ (pdf_loader, text_chunker, embedder, vector_store, retriever, generator) with a narrow, typed interface per module, wired together only by the thin route layer in backend/main.py. The Streamlit app under frontend/ talks to this API purely over HTTP using the requests library; it could be replaced by a different UI, a CLI, or another service entirely without touching any pipeline code."),
      h2("3.2 Per-page chunking (not whole-document chunking)"),
      p("Rather than concatenating the whole PDF into one string before splitting, the chunker runs LangChain's RecursiveCharacterTextSplitter separately on each page's text. This costs a small amount of chunking optimality at page boundaries but guarantees that every chunk can be traced back to exactly one page number, which the assignment's source-attribution requirement depends on."),
      h2("3.3 Namespace-per-knowledge-base, with a backend-side registry"),
      p("Every knowledge base created in the UI maps to one Pinecone namespace inside a single shared index. Because the frontend and backend are now separate processes, tracking which document names belong to which namespace can no longer live in Streamlit's session state \u2014 Pinecone itself has no call to list distinct metadata values. backend/services/registry.py is a small JSON-backed store that solves this on the server side, so any client (not just the bundled Streamlit app) sees a consistent document list per namespace."),
      h2("3.4 Retrieval-grounded confidence score"),
      p("Rather than asking the LLM to self-report a confidence value (which studies show correlates poorly with actual accuracy), confidence is computed from retrieval quality: the average similarity score of the chunks used, scaled down slightly when fewer than three chunks were found. This keeps the score transparent and auditable."),
      h2("3.5 Answer and sources kept structurally separate"),
      p("The API's QueryResponse model returns \u201Canswer\u201D and \u201Csources\u201D as two distinct JSON fields rather than one concatenated string. The frontend renders them in two visually separate sections, so the generated answer and its citations never run together on screen regardless of how long either one is."),

      h1("4. Embedding Model"),
      p("The system uses Sentence-Transformers' all-MiniLM-L6-v2 model, run locally on top of PyTorch (torch), which is listed as a direct, explicit dependency rather than left as an implicit transitive one."),
      bullet("384-dimensional output \u2014 small enough for fast Pinecone queries and low storage cost."),
      bullet("Runs entirely on CPU with no external API call or per-token cost, which matters for a student project without an embedding-API budget. The embedder automatically uses a CUDA GPU instead if one is available (torch.cuda.is_available()), with no code changes needed."),
      bullet("Well-suited to short passages (sentences to paragraphs), which matches the chunk sizes used here (200\u20132000 characters, default 800)."),
      bullet("Embeddings are L2-normalized at generation time so that cosine similarity and dot-product scoring are equivalent, matching Pinecone's cosine metric."),
      p("torchvision is deliberately not included: this pipeline has no image/vision model anywhere in it, only text. It would only become relevant if OCR support for scanned PDFs were added later using a vision-capable model."),

      h1("5. Pinecone Configuration"),
      dataTable(configRows[0], configRows.slice(1), colW),
      p(""),

      h1("6. Hallucination Prevention"),
      p("Hallucination is treated as a pipeline-level concern rather than something the LLM prompt alone can solve, since prompts are not a hard guarantee of behavior:"),
      bullet("Similarity threshold: matches returned by Pinecone below a configurable cosine-similarity cutoff (default 0.35) are discarded before the LLM sees them."),
      bullet("Zero-context short-circuit: if no chunks survive the threshold, generator.py never calls the LLM at all \u2014 it returns the fixed refusal sentence directly in code, removing any chance of the model improvising."),
      bullet("Strict system prompt: the LLM is instructed to answer only from the supplied context and to output the exact sentence \u201CThe answer is not available in the provided document.\u201D when the context is insufficient."),
      bullet("Deterministic generation: temperature is set to 0 to minimize creative drift."),
      bullet("Traceable answers: every non-refused answer carries its source chunks (document name, page number, excerpt, similarity score), so a user can check the claim against the original PDF in seconds."),

      h1("7. Intermediate-Level Enhancements Implemented"),
      p("Seven of the optional enhancements were implemented (the assignment requires at least three):"),
      bullet("Multi-document support \u2014 several PDFs share a namespace and are queried together, with per-document filtering available."),
      bullet("Query history \u2014 both an in-session view and a persisted CSV log (data/logs/query_log.csv)."),
      bullet("Adjustable chunk size and overlap from the sidebar."),
      bullet("Adjustable top-k and similarity threshold from the sidebar."),
      bullet("Metadata filtering \u2014 restrict retrieval to selected documents and/or a page range."),
      bullet("Confidence scoring display \u2014 retrieval-grounded heuristic shown alongside every answer."),
      bullet("Logging user queries \u2014 timestamp, namespace, question, chunks used, confidence, and an answer preview."),

      h1("8. Error Handling"),
      bullet("Invalid PDF: empty files, corrupted files, password-protected files, and image-only PDFs with no text layer each raise a specific InvalidPDFError with an actionable message instead of crashing the app."),
      bullet("Empty queries: retriever.retrieve() rejects blank/whitespace-only questions before any embedding or network call is made."),
      bullet("Pinecone connection failures: all Pinecone calls are wrapped and re-raised as VectorStoreError, which backend/main.py translates into an HTTP 502 response with a readable detail message; the Streamlit frontend then surfaces that message as an error banner instead of a stack trace."),
      bullet("Oversized files: uploads over the configured limit (20 MB default) are rejected with the file's actual size reported back to the user."),

      h1("9. Challenges Faced"),
      p("Preserving accurate page numbers while still getting semantically coherent chunks required chunking per page rather than on the full document text, trading a small amount of cross-page context for reliable source attribution \u2014 this is the main architectural compromise in the system."),
      p("On Windows, installing torch/sentence-transformers from an MSYS2 or Git-Bash shell can fail or behave inconsistently, because that shell's Python is typically an MSYS2/MinGW build rather than the official CPython distribution, and PyTorch's prebuilt wheels only target the latter. The fix documented in the README is simply to use a standard python.org or Anaconda install and run pip/uvicorn/streamlit from a regular terminal instead."),
      p("Splitting the system into a backend API and a separate frontend introduced a genuinely new problem: document-name tracking per namespace, which used to live in Streamlit's session state, had nowhere to live once the frontend became a stateless HTTP client. This is what motivated adding backend/services/registry.py as a small persistent, server-side store."),
      p("Groq deprecated several of its previously standard Llama chat models (including llama-3.1-8b-instant and llama-3.3-70b-versatile) in mid-2026 in favor of its GPT-OSS model family; the LLM model name is therefore read from an environment variable rather than hard-coded, so the deployment can be updated without a code change as Groq's model lineup continues to evolve."),
      p("Choosing a default similarity threshold involved a real precision/recall trade-off: too low and irrelevant chunks reach the LLM (hallucination risk), too high and legitimate answers get refused unnecessarily. 0.35 was chosen as a middle-ground default and exposed as a UI slider so it can be tuned per document set."),

      h1("10. Performance Analysis"),
      p("Indexing throughput is dominated by local embedding generation rather than network calls: all-MiniLM-L6-v2 encodes roughly 100\u2013300 short chunks per second on a modern CPU (batch size 32), meaning a typical 20-page policy document (roughly 150\u2013250 chunks at the default settings) embeds in a few seconds before the Pinecone upsert."),
      p("Query latency is dominated by the LLM call rather than retrieval: Pinecone serverless queries typically return in well under 200 ms, while Groq's LPU-based inference is comparatively fast for its class \u2014 published rates are on the order of hundreds of tokens per second for small open models such as GPT-OSS-20B, keeping end-to-end answer generation in the low single-digit seconds for typical answer lengths."),
      p("Cost is intentionally kept low for a student deployment: embeddings are free and local, Pinecone's serverless free tier comfortably covers a document set of this scale, and Groq's smaller GPT-OSS-20B route is priced at a fraction of a cent per typical query, making iterative testing inexpensive."),

      h1("11. Conclusion and Future Work"),
      p("The system satisfies the required architecture end-to-end and demonstrates the mandated Pinecone features (index creation, namespaces, upserting, querying, and metadata management) alongside seven optional enhancements. Natural next steps would be adding OCR for scanned PDFs, hybrid dense+sparse retrieval for better exact-match recall on names and figures, a cross-encoder reranking stage before generation, streaming LLM responses in the UI, and automated regression tests around the hallucination guardrails described in Section 6."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(__dirname + "/technical_report.docx", buf);
  console.log("Report written:", buf.length, "bytes");
});
