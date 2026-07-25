/* =============================================================
   RAG Dashboard — Application Logic
   Every number on this dashboard is either read directly from the
   backend or computed from real data the backend returned (session
   history, query logs, document records). Nothing here is a mock/
   placeholder value, per the project's own no-hallucination principle
   -- it would be a bit rich for the surrounding dashboard to fake stats
   while the RAG system itself refuses to make things up.
   ============================================================= */
(() => {
  const API = ""; // same-origin: this page is served by the backend
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const PIPELINE_STAGES = [
    { key: "upload",   icon: "fa-cloud-arrow-up",   label: "PDF Upload" },
    { key: "extract",  icon: "fa-file-lines",       label: "Text Extraction" },
    { key: "chunk",    icon: "fa-puzzle-piece",     label: "Chunking" },
    { key: "embed",    icon: "fa-vector-square",    label: "Embedding" },
    { key: "index",    icon: "fa-cube",             label: "Pinecone Upsert" },
    { key: "retrieve", icon: "fa-magnifying-glass", label: "Retrieval" },
    { key: "generate", icon: "fa-wand-magic-sparkles", label: "LLM Answer" },
    { key: "cite",     icon: "fa-quote-right",      label: "Source Attribution" },
  ];

  const EXAMPLE_QUESTIONS = [
    "What is this document about?",
    "Summarize the key points",
    "What are the important dates or deadlines?",
    "Are there any numeric limits or thresholds mentioned?",
  ];

  const state = {
    namespace: localStorage.getItem("rag_namespace") || ("session-" + Math.random().toString(16).slice(2, 10)),
    ocrMode: "auto",
    chunkSize: 800, chunkOverlap: 120, topK: 5, threshold: 0.25,
    documents: [],       // real DocumentInfo[] from /api/documents
    docFilter: new Set(),
    sessionHistory: [],  // real answers received this session
    serverLogs: [],      // real rows from /api/logs
    health: null,
    vectorSnapshots: [], // real (timestamp, totalVectors) pairs, taken as documents change this session
    pageCiteCounts: {},  // real "doc.pdf p.N" -> citation count, this session
    charts: {},
  };

  // ---------------------------------------------------------------- utils
  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function fmtDate(iso) {
    if (!iso) return "unknown";
    try { return new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
    catch { return iso; }
  }
  function totalVectors() {
    return state.documents.reduce((sum, d) => sum + (d.chunks || 0), 0);
  }
  function snapshotVectors() {
    state.vectorSnapshots.push({ t: Date.now(), v: totalVectors() });
    if (state.vectorSnapshots.length > 40) state.vectorSnapshots.shift();
  }

  // ---------------------------------------------------------------- toasts
  function toast(type, message) {
    const icons = { ok: "fa-circle-check", err: "fa-circle-xmark", info: "fa-circle-info" };
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i><span>${esc(message)}</span>`;
    $("#toast-stack").appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; el.style.transform = "translateX(30px)"; el.style.transition = "all .3s"; setTimeout(() => el.remove(), 300); }, 4200);
  }

  // ---------------------------------------------------------------- ripple
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".btn, .icon-btn, .chip");
    if (!btn) return;
    const r = document.createElement("span");
    r.className = "ripple";
    const rect = btn.getBoundingClientRect();
    r.style.left = (e.clientX - rect.left - 6) + "px";
    r.style.top = (e.clientY - rect.top - 6) + "px";
    r.style.width = r.style.height = "12px";
    btn.style.position = btn.style.position || "relative";
    btn.style.overflow = "hidden";
    btn.appendChild(r);
    setTimeout(() => r.remove(), 600);
  });

  // ---------------------------------------------------------------- api
  async function apiGet(path, params) {
    const url = new URL(API + path, window.location.origin);
    if (params) Object.entries(params).forEach(([k, v]) => v !== undefined && v !== null && url.searchParams.set(k, v));
    return fetch(url);
  }
  async function apiPostJSON(path, body) {
    return fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  }
  async function apiPostForm(path, formData) {
    return fetch(API + path, { method: "POST", body: formData });
  }
  async function apiDelete(path, params) {
    const url = new URL(API + path, window.location.origin);
    if (params) Object.entries(params).forEach(([k, v]) => v !== undefined && url.searchParams.set(k, v));
    return fetch(url, { method: "DELETE" });
  }
  async function errorDetail(resp) {
    try { const j = await resp.json(); return j.detail || resp.statusText; }
    catch { return resp.statusText || ("HTTP " + resp.status); }
  }

  // ---------------------------------------------------------------- theme
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("rag_theme", theme);
  }
  applyTheme(localStorage.getItem("rag_theme") || "dark");
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    applyTheme(next);
  });

  // ---------------------------------------------------------------- sidebar
  const sidebar = $("#sidebar");
  $("#sidebar-toggle").addEventListener("click", () => sidebar.classList.toggle("collapsed"));
  $("#mobile-menu-btn").addEventListener("click", () => sidebar.classList.toggle("mobile-open"));

  // ---------------------------------------------------------------- views
  const VIEW_META = {
    dashboard: ["Dashboard", "Live overview of your RAG pipeline"],
    upload: ["Upload Document", "Add a PDF to your knowledge base"],
    library: ["Document Library", "Everything indexed in this namespace"],
    ask: ["Ask Questions", "Grounded answers, traced to their source"],
    history: ["Retrieval History", "Every question asked, saved and searchable"],
    analytics: ["Analytics", "Real usage patterns from this deployment"],
    settings: ["Settings", "Retrieval and generation configuration"],
    about: ["About Project", "System status and API reference"],
  };

  function goToView(name) {
    $$(".view").forEach(v => v.classList.remove("active"));
    const target = $("#view-" + name);
    if (!target) return;
    target.classList.add("active");
    $$(".nav-item").forEach(n => n.classList.toggle("active", n.dataset.view === name));
    const [title, sub] = VIEW_META[name] || [name, ""];
    $("#view-title").textContent = title;
    $("#view-subtitle").textContent = sub;
    sidebar.classList.remove("mobile-open");
    if (name === "library") renderDocLibrary();
    if (name === "history") renderHistory();
    if (name === "analytics") renderAnalytics();
    if (name === "about") renderStatusGrid();
    if (window.AOS) setTimeout(() => AOS.refreshHard(), 60);
  }
  $$(".nav-item").forEach(item => item.addEventListener("click", () => goToView(item.dataset.view)));
  $$("[data-goto]").forEach(el => el.addEventListener("click", () => goToView(el.dataset.goto)));

  // ---------------------------------------------------------------- namespace (kept in sync across 3 inputs)
  const nsInputs = [$("#namespace-input"), $("#namespace-input-upload"), $("#namespace-input-settings")];
  nsInputs.forEach(el => { el.value = state.namespace; });
  nsInputs.forEach(el => el.addEventListener("change", () => {
    state.namespace = el.value.trim() || state.namespace;
    localStorage.setItem("rag_namespace", state.namespace);
    nsInputs.forEach(other => other.value = state.namespace);
    renderPills(); refreshDocuments();
  }));

  function renderPills() {
    // connection dots live in the sidebar footer; document count badge in nav
    $("#nav-doc-count").textContent = state.documents.length;
  }

  // ---------------------------------------------------------------- range sliders (value label + fill)
  function bindRange(id, labelId, key, fmt) {
    const el = $("#" + id);
    const label = labelId ? $("#" + labelId) : null;
    const update = () => {
      const min = parseFloat(el.min), max = parseFloat(el.max), val = parseFloat(el.value);
      el.style.setProperty("--pct", ((val - min) / (max - min) * 100) + "%");
      state[key] = val;
      if (label) label.textContent = fmt ? fmt(val) : val;
    };
    el.addEventListener("input", update);
    update();
  }
  bindRange("chunk-size", "chunk-size-val", "chunkSize", v => v + " chars");
  bindRange("chunk-overlap", "chunk-overlap-val", "chunkOverlap", v => v + " chars");
  bindRange("top-k", "top-k-val", "topK", v => v);
  bindRange("threshold", "threshold-val", "threshold", v => v.toFixed(2));

  $$("#ocr-mode-upload button").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#ocr-mode-upload button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.ocrMode = btn.dataset.val;
    });
  });

  // ---------------------------------------------------------------- KPI count-up
  function countUp(el, target, suffix = "") {
    const start = parseFloat(el.dataset.count) || 0;
    const duration = 900;
    const t0 = performance.now();
    function step(now) {
      const p = Math.min((now - t0) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = start + (target - start) * eased;
      el.textContent = (suffix === "%" ? Math.round(val) : Math.round(val)) + suffix;
      if (p < 1) requestAnimationFrame(step); else el.dataset.count = target;
    }
    requestAnimationFrame(step);
  }

  function renderKPIs() {
    const docs = state.documents.length;
    const chunks = totalVectors();
    const logsForNs = state.serverLogs.filter(l => l.namespace === state.namespace);
    const queries = logsForNs.length;
    const grounded = logsForNs.filter(l => l.used_context === "True");
    const avgConf = grounded.length
      ? grounded.reduce((s, l) => s + (parseFloat(l.confidence) || 0), 0) / grounded.length
      : 0;

    countUp($("#kpi-docs"), docs);
    countUp($("#kpi-chunks"), chunks);
    countUp($("#kpi-vectors"), chunks); // 1 chunk = 1 vector in this pipeline, genuinely equal
    countUp($("#kpi-queries"), queries);
    countUp($("#kpi-confidence"), Math.round(avgConf * 100), "%");
  }

  // ---------------------------------------------------------------- pipeline visual
  function renderPipeline() {
    const el = $("#pipeline-visual");
    el.innerHTML = PIPELINE_STAGES.map((s, i) => `
      <div class="pipe-stage" data-stage="${s.key}">
        ${i > 0 ? '<div class="pipe-connector"></div>' : ""}
        <div class="pipe-node"><i class="fa-solid ${s.icon}"></i></div>
        <div class="pipe-label">${s.label}</div>
        <div class="pipe-sub">Stage ${i + 1}</div>
      </div>`).join("");
  }
  function animatePipeline(stageKeys, stepDelay = 380) {
    const stages = $$(".pipe-stage");
    stages.forEach(s => s.classList.remove("active", "done"));
    stageKeys.forEach((key, i) => {
      setTimeout(() => {
        const el = $(`.pipe-stage[data-stage="${key}"]`);
        if (!el) return;
        stages.forEach(s => s.classList.remove("active"));
        el.classList.add("active");
        el.classList.add("done");
        // mark all previous as done too
        const idx = PIPELINE_STAGES.findIndex(s => s.key === key);
        PIPELINE_STAGES.slice(0, idx).forEach(s => $(`.pipe-stage[data-stage="${s.key}"]`)?.classList.add("done"));
        if (i === stageKeys.length - 1) {
          setTimeout(() => el.classList.remove("active"), 700);
        }
      }, i * stepDelay);
    });
  }

  // ---------------------------------------------------------------- init / health
  async function init() {
    renderPipeline();
    try {
      const r = await apiGet("/api/health");
      state.health = await r.json();
    } catch {
      toast("err", "Can't reach the backend API.");
      $(".main").innerHTML = `<div class="empty-state"><i class="fa-solid fa-plug-circle-xmark"></i><p>Can't reach the backend. Make sure the FastAPI server is running and you're viewing this page from the same origin.</p></div>`;
      return;
    }
    renderConnectionDots();
    await refreshDocuments();
    await refreshLogs();
    renderKPIs();
    renderExampleChips();
    if (window.AOS) AOS.init({ duration: 600, once: true, offset: 30 });
  }

  function renderConnectionDots() {
    const h = state.health;
    $("#dot-pinecone").className = "conn-dot " + (h.pinecone_connected ? "on" : "off");
    $("#dot-groq").className = "conn-dot " + (h.groq_configured ? "on" : "off");
    $("#dot-ocr").className = "conn-dot " + (h.ocr_available ? "on" : "off");
  }

  async function refreshDocuments() {
    const r = await apiGet("/api/documents", { namespace: state.namespace });
    const body = r.ok ? await r.json() : { documents: [] };
    state.documents = body.documents || [];
    snapshotVectors();
    renderPills();
    renderKPIs();
    renderDocFilterList();
    if ($("#view-library").classList.contains("active")) renderDocLibrary();
  }

  // ---------------------------------------------------------------- doc filter (Ask view)
  function renderDocFilterList() {
    const el = $("#doc-filter-list");
    if (!state.documents.length) {
      el.innerHTML = `<span class="field-hint">Upload a document to enable filtering.</span>`;
      return;
    }
    el.innerHTML = state.documents.map(d => `
      <label style="display:flex; gap:8px; align-items:baseline; cursor:pointer;">
        <input type="checkbox" data-doc="${esc(d.name)}" ${state.docFilter.has(d.name) ? "checked" : ""} style="accent-color:var(--brand-indigo);">
        <span style="overflow-wrap:anywhere;">${esc(d.name)}</span>
      </label>`).join("");
    $$("#doc-filter-list input[type=checkbox]").forEach(cb => {
      cb.addEventListener("change", () => {
        cb.checked ? state.docFilter.add(cb.dataset.doc) : state.docFilter.delete(cb.dataset.doc);
      });
    });
  }

  // ---------------------------------------------------------------- UPLOAD
  const dropzone = $("#dropzone"), fileInput = $("#file-input");
  dropzone.addEventListener("click", () => fileInput.click());
  ["dragenter", "dragover"].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", e => stageFiles(e.dataTransfer.files));
  fileInput.addEventListener("change", e => stageFiles(e.target.files));

  let stagedFiles = [];
  function stageFiles(fileList) {
    const files = Array.from(fileList).filter(f => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
    if (!files.length) { toast("err", "Only PDF files are supported."); return; }
    stagedFiles = files;
    renderFileList();
  }
  function renderFileList() {
    $("#file-list").innerHTML = stagedFiles.map((f, i) => `
      <div class="card file-card" data-idx="${i}">
        <div class="f-icon"><i class="fa-solid fa-file-pdf"></i></div>
        <div class="f-info">
          <div class="f-name">${esc(f.name)}</div>
          <div class="f-meta">${(f.size / 1024 / 1024).toFixed(2)} MB</div>
        </div>
        <div class="f-status pending">ready</div>
      </div>`).join("");
  }

  $("#upload-btn").addEventListener("click", async () => {
    if (!stagedFiles.length) { toast("err", "Choose or drop a PDF first."); return; }
    for (let i = 0; i < stagedFiles.length; i++) {
      const f = stagedFiles[i];
      const row = $(`.file-card[data-idx="${i}"] .f-status`);
      row.className = "f-status pending"; row.innerHTML = `<span class="spinner"></span>`;
      animatePipeline(["upload", "extract", "chunk", "embed", "index"], 420);

      const form = new FormData();
      form.append("file", f);
      form.append("namespace", state.namespace);
      form.append("chunk_size", state.chunkSize);
      form.append("chunk_overlap", state.chunkOverlap);
      form.append("ocr_mode", state.ocrMode);

      try {
        const r = await apiPostForm("/api/documents/upload", form);
        if (r.ok) {
          const b = await r.json();
          const ocrNote = b.pages_via_ocr ? ` · ${b.pages_via_ocr} via OCR` : "";
          row.className = "f-status ok";
          row.textContent = `${b.pages_extracted}p · ${b.chunks_indexed} chunks${ocrNote}`;
          toast("ok", `${f.name} indexed (${b.chunks_indexed} chunks)`);
        } else {
          row.className = "f-status err";
          row.textContent = "failed";
          toast("err", `${f.name}: ${await errorDetail(r)}`);
        }
      } catch {
        row.className = "f-status err"; row.textContent = "network error";
        toast("err", `${f.name}: network error reaching backend`);
      }
    }
    stagedFiles = [];
    fileInput.value = "";
    await refreshDocuments();
  });

  // ---------------------------------------------------------------- LIBRARY
  function renderDocLibrary() {
    const grid = $("#doc-grid");
    if (!state.documents.length) {
      grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;"><i class="fa-solid fa-folder-open"></i><p>Nothing indexed yet in this namespace. Upload a PDF to get started.</p></div>`;
      return;
    }
    grid.innerHTML = state.documents.map(d => `
      <div class="card hoverable doc-card">
        <div class="dc-top">
          <div class="dc-icon"><i class="fa-solid fa-file-pdf"></i></div>
          <div style="flex:1; min-width:0;">
            <div class="dc-name">${esc(d.name)}</div>
            <div class="dc-date">Uploaded ${fmtDate(d.uploaded_at)}</div>
          </div>
          ${d.ocr_pages ? `<span class="status-badge ocr">OCR</span>` : `<span class="status-badge text">TEXT</span>`}
        </div>
        <div class="dc-stats">
          <span><b>${d.pages}</b> pages</span>
          <span><b>${d.chunks}</b> chunks</span>
          ${d.ocr_pages ? `<span><b>${d.ocr_pages}</b> via OCR</span>` : ""}
        </div>
        <div class="dc-actions">
          <button class="btn btn-ghost btn-sm" data-reindex="${esc(d.name)}"><i class="fa-solid fa-rotate"></i> Re-upload to update</button>
          <button class="btn btn-danger btn-sm" data-delete="${esc(d.name)}"><i class="fa-solid fa-trash"></i></button>
        </div>
      </div>`).join("");

    $$("#doc-grid [data-delete]").forEach(btn => btn.addEventListener("click", async () => {
      const name = btn.dataset.delete;
      if (!confirm(`Delete "${name}" from this namespace? This removes its vectors from Pinecone and can't be undone.`)) return;
      const r = await apiDelete(`/api/documents/${encodeURIComponent(name)}`, { namespace: state.namespace });
      if (r.ok) { toast("ok", `${name} deleted.`); await refreshDocuments(); renderDocLibrary(); }
      else toast("err", await errorDetail(r));
    }));
    $$("#doc-grid [data-reindex]").forEach(btn => btn.addEventListener("click", () => {
      goToView("upload");
      toast("info", `Upload a new version of "${btn.dataset.reindex}" — it will replace the existing record.`);
    }));
  }
  $("#refresh-docs-btn").addEventListener("click", async () => { await refreshDocuments(); renderDocLibrary(); toast("ok", "Refreshed."); });

  // ---------------------------------------------------------------- ASK / CHAT
  function renderExampleChips() {
    $("#example-chips").innerHTML = EXAMPLE_QUESTIONS.map(q => `<span class="chip">${esc(q)}</span>`).join("");
    $$("#example-chips .chip").forEach(chip => chip.addEventListener("click", () => {
      $("#question-input").value = chip.textContent;
      runAsk();
    }));
  }

  function appendMsg(role, html, extraClass = "") {
    $("#chat-empty")?.remove();
    const log = $("#chat-log");
    const wrap = document.createElement("div");
    wrap.className = `msg ${role} ${extraClass}`.trim();
    wrap.innerHTML = `
      <div class="avatar"><i class="fa-solid ${role === "user" ? "fa-user" : "fa-sparkles"}"></i></div>
      <div class="bubble">${html}</div>`;
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return wrap;
  }

  $("#ask-btn").addEventListener("click", runAsk);
  $("#question-input").addEventListener("keydown", e => { if (e.key === "Enter") runAsk(); });
  $("#clear-chat-btn").addEventListener("click", () => {
    $("#chat-log").innerHTML = `<div class="empty-state" id="chat-empty"><i class="fa-solid fa-comments"></i><p>No questions yet. Try one of the examples below, or type your own.</p></div>`;
  });

  async function runAsk() {
    const q = $("#question-input").value.trim();
    if (!q) { toast("err", "Type a question first."); return; }
    if (!state.documents.length) { toast("err", "Upload a document before asking questions."); return; }

    appendMsg("user", esc(q));
    $("#question-input").value = "";
    const typing = appendMsg("assistant", `<span class="typing-dots"><span></span><span></span><span></span></span>`);
    animatePipeline(["retrieve", "generate", "cite"], 500);

    const payload = {
      question: q,
      namespace: state.namespace,
      top_k: state.topK,
      similarity_threshold: state.threshold,
      document_filter: state.docFilter.size ? Array.from(state.docFilter) : null,
      page_min: null, page_max: null,
    };

    let r;
    try { r = await apiPostJSON("/api/query", payload); }
    catch { typing.remove(); appendMsg("assistant", "Network error reaching the backend.", "absent"); return; }

    if (!r.ok) { typing.remove(); appendMsg("assistant", esc(await errorDetail(r)), "absent"); return; }
    const result = await r.json();
    typing.remove();

    const grounded = result.used_context;
    let html = `<span class="eyebrow">${grounded ? "Grounded answer" : "Not found in these documents"}</span>${esc(result.answer)}`;
    const bubbleEl = appendMsg("assistant", html, grounded ? "" : "absent");

    if (grounded) {
      const conf = Math.round(result.confidence * 100);
      const meterHtml = `
        <div class="answer-meter"><div class="meter-track"><div class="meter-fill" style="width:${conf}%"></div></div>
        <div class="meter-label">confidence ${conf}% · ${result.sources.length} passage(s)</div></div>`;
      let exhibitsHtml = "";
      result.sources.forEach((s, i) => {
        const score = Math.round(s.similarity_score * 100);
        const key = `${s.document_name} p.${s.page_number}`;
        state.pageCiteCounts[key] = (state.pageCiteCounts[key] || 0) + 1;
        exhibitsHtml += `
          <div class="exhibit">
            <div class="ex-head"><span class="ex-stamp">${i + 1} · PAGE ${s.page_number}</span><span class="ex-doc">${esc(s.document_name)}</span></div>
            <div class="ex-excerpt">${esc(s.excerpt)}</div>
            <div class="ex-chunkid">chunk_id: ${esc(s.chunk_id)}</div>
            <div class="answer-meter"><div class="meter-track"><div class="meter-fill" style="width:${score}%"></div></div>
            <div class="meter-label">similarity ${s.similarity_score.toFixed(3)}</div></div>
          </div>`;
      });
      const actionsHtml = `
        <div class="answer-actions">
          <button class="btn btn-ghost btn-sm" data-copy-answer><i class="fa-solid fa-copy"></i> Copy</button>
          <button class="btn btn-ghost btn-sm" data-download-answer><i class="fa-solid fa-download"></i> Download</button>
        </div>`;
      bubbleEl.querySelector(".bubble").insertAdjacentHTML("beforeend", meterHtml + actionsHtml + exhibitsHtml);
      bubbleEl.querySelector("[data-copy-answer]").addEventListener("click", () => {
        navigator.clipboard.writeText(result.answer).then(() => toast("ok", "Answer copied."));
      });
      bubbleEl.querySelector("[data-download-answer]").addEventListener("click", () => {
        const blob = new Blob([`Q: ${q}\n\nA: ${result.answer}\n\nSources:\n` +
          result.sources.map(s => `- ${s.document_name} p.${s.page_number} (chunk ${s.chunk_id}, similarity ${s.similarity_score.toFixed(3)})`).join("\n")],
          { type: "text/plain" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = "answer.txt"; a.click();
      });
    }
    $("#chat-log").scrollTop = $("#chat-log").scrollHeight;

    state.sessionHistory.unshift({
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      question: q, answer: result.answer, grounded, confidence: result.confidence,
      sources: result.sources.map(s => `${s.document_name} p.${s.page_number} (${s.similarity_score.toFixed(2)})`),
    });
    await refreshLogs();
    renderKPIs();
  }

  // ---------------------------------------------------------------- HISTORY
  async function refreshLogs() {
    const r = await apiGet("/api/logs", { limit: 200 });
    state.serverLogs = r.ok ? await r.json() : [];
  }
  function renderHistory() {
    const el = $("#history-list");
    const q = ($("#history-search").value || "").toLowerCase();
    const rows = state.serverLogs.filter(l => !q || l.question.toLowerCase().includes(q));
    if (!rows.length) {
      el.innerHTML = `<div class="empty-state"><i class="fa-solid fa-inbox"></i><p>No logged queries yet.</p></div>`;
      return;
    }
    el.innerHTML = rows.map(l => `
      <div class="history-item">
        <div class="history-time">${l.timestamp_utc.slice(11, 16)}</div>
        <div class="history-body">
          <div class="history-q">${esc(l.question)}</div>
          <div class="history-a">${esc(l.answer_preview)} · confidence ${Math.round((parseFloat(l.confidence) || 0) * 100)}% · ns: ${esc(l.namespace)}</div>
        </div>
      </div>`).join("");
  }
  $("#history-search").addEventListener("input", renderHistory);

  // ---------------------------------------------------------------- ANALYTICS
  function renderAnalytics() {
    // --- queries per day + confidence trend (real, from server logs) ---
    const byDay = {};
    state.serverLogs.forEach(l => {
      const day = (l.timestamp_utc || "").slice(0, 10);
      if (!day) return;
      byDay[day] = byDay[day] || { count: 0, confSum: 0, confN: 0 };
      byDay[day].count++;
      if (l.used_context === "True") { byDay[day].confSum += parseFloat(l.confidence) || 0; byDay[day].confN++; }
    });
    const days = Object.keys(byDay).sort();
    const counts = days.map(d => byDay[d].count);
    const confs = days.map(d => byDay[d].confN ? Math.round(100 * byDay[d].confSum / byDay[d].confN) : null);

    renderChart("chart-queries", "bar", {
      labels: days.length ? days : ["No data yet"],
      datasets: [
        { type: "bar", label: "Queries", data: days.length ? counts : [0], backgroundColor: "rgba(99,102,241,.55)", borderRadius: 6, yAxisID: "y" },
        { type: "line", label: "Avg confidence %", data: days.length ? confs : [0], borderColor: "#22d3ee", backgroundColor: "transparent", tension: .35, yAxisID: "y1" },
      ],
    }, { scales: { y: { position: "left", grid: { color: "rgba(255,255,255,.06)" } }, y1: { position: "right", min: 0, max: 100, grid: { display: false } } } });

    // --- top cited pages (real, this session only -- server logs don't store page numbers) ---
    const topPages = Object.entries(state.pageCiteCounts).sort((a, b) => b[1] - a[1]).slice(0, 6);
    const maxCite = Math.max(1, ...topPages.map(p => p[1]));
    $("#top-pages-list").innerHTML = topPages.length
      ? topPages.map(([page, n]) => `
          <div class="top-page-row"><span style="width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${esc(page)}">${esc(page)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${(n/maxCite*100)}%"></div></div><span class="bar-val">${n}</span></div>`).join("")
      : `<span class="field-hint">No citations yet this session — ask a question to populate this.</span>`;

    // --- document activity: chunks per document (real, from registry) ---
    renderChart("chart-docs", "bar", {
      labels: state.documents.length ? state.documents.map(d => d.name) : ["No documents yet"],
      datasets: [{ label: "Chunks indexed", data: state.documents.length ? state.documents.map(d => d.chunks) : [0], backgroundColor: "rgba(34,211,238,.5)", borderRadius: 6 }],
    });

    // --- vector growth this session (real snapshots) ---
    renderChart("chart-vectors", "line", {
      labels: state.vectorSnapshots.map((s, i) => "T+" + i),
      datasets: [{ label: "Total vectors", data: state.vectorSnapshots.map(s => s.v), borderColor: "#8b5cf6", backgroundColor: "rgba(139,92,246,.15)", fill: true, tension: .3 }],
    });
  }

  function renderChart(canvasId, type, data, extraOptions = {}) {
    const ctx = $("#" + canvasId);
    if (!ctx || !window.Chart) return;
    if (state.charts[canvasId]) state.charts[canvasId].destroy();
    const isDark = document.documentElement.getAttribute("data-theme") !== "light";
    const gridColor = isDark ? "rgba(255,255,255,.06)" : "rgba(0,0,0,.06)";
    const tickColor = isDark ? "#8b93b0" : "#5c6584";
    state.charts[canvasId] = new Chart(ctx, {
      type, data,
      options: Object.assign({
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: tickColor, font: { size: 11 } } } },
        scales: { x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 10 } } },
                  y: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 10 } } } },
      }, extraOptions),
    });
  }

  // ---------------------------------------------------------------- SETTINGS
  $("#clear-namespace-btn").addEventListener("click", async () => {
    if (!confirm(`Delete every indexed passage in namespace "${state.namespace}"? This can't be undone.`)) return;
    const r = await apiDelete(`/api/namespaces/${encodeURIComponent(state.namespace)}`);
    if (r.ok) { toast("ok", "Namespace cleared."); await refreshDocuments(); renderDocLibrary(); }
    else toast("err", await errorDetail(r));
  });

  // ---------------------------------------------------------------- ABOUT / system status
  function renderStatusGrid() {
    const h = state.health;
    if (!h) return;
    const rows = [
      { name: "Pinecone", ok: h.pinecone_connected, detail: h.pinecone_connected ? "connected" : "not connected" },
      { name: "Groq LLM", ok: h.groq_configured, detail: h.groq_configured ? "API key configured" : "missing API key" },
      { name: "OCR (Tesseract)", ok: h.ocr_available, detail: h.ocr_status },
      { name: "Embedding model", ok: true, detail: h.embedding_model + " · " + h.embedding_device },
    ];
    $("#status-grid").innerHTML = rows.map(r => `
      <div class="card status-card">
        <span class="st-dot" style="background:${r.ok ? "var(--success)" : "var(--danger)"}"></span>
        <div><div class="st-name">${esc(r.name)}</div><div class="st-detail">${esc(r.detail)}</div></div>
      </div>`).join("");
  }

  init();
})();
