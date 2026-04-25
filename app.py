import streamlit as st
import os
import time
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Enterprise RAG Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - dark editorial aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --accent: #7c6aff;
    --accent2: #ff6a6a;
    --accent3: #6affd4;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --border: #2a2a3a;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    color: var(--text);
}

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

/* Buttons */
.stButton > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 4px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #9580ff;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124,106,255,0.4);
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface2);
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 1rem;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Answer box */
.answer-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 4px 4px 0;
    padding: 1.5rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Citation card */
.citation-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.8rem;
    color: var(--muted);
}
.citation-card .source-tag {
    display: inline-block;
    background: var(--accent);
    color: white;
    font-size: 0.65rem;
    padding: 2px 6px;
    border-radius: 2px;
    margin-right: 6px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.citation-card.csv-source .source-tag { background: var(--accent3); color: #0a0a0f; }
.citation-card.web-source .source-tag { background: var(--accent2); }

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem;
    text-align: center;
}
.metric-card .val { font-size: 1.5rem; font-family: 'Syne', sans-serif; font-weight: 700; color: var(--accent); }
.metric-card .lbl { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #111118 0%, #1a1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(124,106,255,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.header-banner h1 { margin: 0; font-size: 1.8rem; }
.header-banner p { margin: 0.4rem 0 0; color: var(--muted); font-size: 0.8rem; letter-spacing: 0.05em; }

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    font-family: 'Syne', sans-serif;
}
.status-ready { background: rgba(106,255,212,0.15); color: var(--accent3); border: 1px solid rgba(106,255,212,0.3); }
.status-pending { background: rgba(255,106,106,0.15); color: var(--accent2); border: 1px solid rgba(255,106,106,0.3); }

/* Divider */
hr { border-color: var(--border) !important; }

/* Info/warning boxes */
.stInfo, .stWarning, .stSuccess, .stError {
    border-radius: 4px !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Chat messages */
.chat-message {
    padding: 1rem;
    border-radius: 4px;
    margin: 0.5rem 0;
}
.user-msg {
    background: var(--surface2);
    border-left: 3px solid var(--accent2);
}
.bot-msg {
    background: var(--surface);
    border-left: 3px solid var(--accent);
}
</style>
""", unsafe_allow_html=True)

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    st.markdown("### Model")
    model = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], label_visibility="collapsed")
    chunk_size = st.slider("Chunk size", 256, 1024, 512, 64)
    top_k = st.slider("Top-K retrieval", 2, 10, 5)
    use_rerank = st.toggle("Enable Re-Ranking", value=True)

    st.divider()
    st.markdown("### Retriever Weights")
    w_vec = st.slider("Vector Search", 0.0, 1.0, 0.5, 0.05)
    w_sw  = st.slider("Sentence Window", 0.0, 1.0, 0.3, 0.05)
    w_kw  = st.slider("Keyword (BM25)", 0.0, 1.0, 0.2, 0.05)

    st.divider()

    # Index status
    badge = '<span class="status-badge status-ready">● READY</span>' if st.session_state.index_ready \
            else '<span class="status-badge status-pending">● NOT INDEXED</span>'
    st.markdown(f"**Index Status** {badge}", unsafe_allow_html=True)

    if st.session_state.ingested_sources:
        st.markdown("**Sources:**")
        for s in st.session_state.ingested_sources:
            st.markdown(f"- `{s}`")

    if st.button("🗑️ Clear Index & History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.index_ready = False
        st.session_state.ingested_sources = []
        st.session_state.query_count = 0
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🧠 Enterprise RAG Knowledge Base</h1>
    <p>MULTI-SOURCE · MULTI-RETRIEVER · RE-RANKED · CITED</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    src_count = len(st.session_state.ingested_sources)
    st.metric("Sources Indexed", src_count)
with col2:
    st.metric("Queries Run", st.session_state.query_count)
with col3:
    st.metric("Retrievers Active", 3)
with col4:
    st.metric("Re-Ranking", "ON" if use_rerank else "OFF")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ingest, tab_query, tab_history = st.tabs(["📥  Data Ingestion", "🔍  Query", "💬  History"])

# ── TAB 1: INGEST ─────────────────────────────────────────────────────────────
with tab_ingest:
    st.markdown("### Ingest Your Data Sources")
    st.markdown("Upload files or point to a web URL. All sources are merged into a unified vector index.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### 📄 PDF Documents")
        pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_upload")

        st.markdown("#### 📊 CSV / Structured Data")
        csv_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True, key="csv_upload")

    with col_r:
        st.markdown("#### 🌐 Website URL")
        web_url = st.text_input("Enter website URL to scrape", placeholder="https://example.com/docs")
        max_pages = st.slider("Max pages to crawl", 1, 20, 5)

        st.markdown("#### 📝 Plain Text")
        raw_text = st.text_area("Paste text directly", height=100, placeholder="Paste any text content here...")

    st.divider()
    if st.button("⚡ Build Index", use_container_width=True, type="primary"):
        if not openai_key:
            st.error("Please enter your OpenAI API key in the sidebar first.")
        else:
            has_content = pdf_files or csv_files or web_url.strip() or raw_text.strip()
            if not has_content:
                st.warning("Please provide at least one data source.")
            else:
                try:
                    from rag_engine import RAGEngine
                    with st.status("⚙️ Building index...", expanded=True) as status:
                        engine = RAGEngine(
                            openai_api_key=openai_key,
                            model=model,
                            chunk_size=chunk_size,
                            top_k=top_k,
                            use_rerank=use_rerank,
                            weights=(w_vec, w_sw, w_kw)
                        )

                        sources = []

                        if pdf_files:
                            st.write("📄 Loading PDFs...")
                            engine.ingest_pdfs(pdf_files)
                            for f in pdf_files:
                                sources.append(f"PDF: {f.name}")

                        if csv_files:
                            st.write("📊 Loading CSVs...")
                            engine.ingest_csvs(csv_files)
                            for f in csv_files:
                                sources.append(f"CSV: {f.name}")

                        if web_url.strip():
                            st.write(f"🌐 Scraping {web_url}...")
                            engine.ingest_web(web_url.strip(), max_pages)
                            sources.append(f"Web: {web_url.strip()[:40]}...")

                        if raw_text.strip():
                            st.write("📝 Loading text...")
                            engine.ingest_text(raw_text.strip(), "manual_input")
                            sources.append("Text: manual input")

                        st.write("🔨 Chunking & embedding...")
                        engine.build_index()

                        st.session_state.engine = engine
                        st.session_state.index_ready = True
                        st.session_state.ingested_sources = sources
                        status.update(label="✅ Index built successfully!", state="complete")

                    st.success(f"Indexed {len(sources)} source(s). Switch to the **Query** tab to start asking questions.")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error building index: {e}")

# ── TAB 2: QUERY ─────────────────────────────────────────────────────────────
with tab_query:
    st.markdown("### Ask Your Knowledge Base")

    if not st.session_state.index_ready:
        st.info("👆 Go to the **Data Ingestion** tab first to build your index.")
    else:
        query = st.text_input("Your question", placeholder="What does the document say about...")
        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            submit = st.button("🔍 Search & Answer", type="primary", use_container_width=True)
        with col_q2:
            show_ctx = st.toggle("Show context", value=True)

        if submit and query.strip():
            st.session_state.query_count += 1
            engine = st.session_state.engine

            with st.spinner("Retrieving, re-ranking, and generating..."):
                t0 = time.time()
                try:
                    result = engine.query(query.strip())
                    elapsed = time.time() - t0

                    # Save to history
                    st.session_state.messages.append({
                        "query": query.strip(),
                        "answer": result["answer"],
                        "sources": result.get("sources", []),
                        "latency": elapsed
                    })

                    # Answer
                    st.markdown("#### 💡 Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                    # Latency
                    st.caption(f"⏱ {elapsed:.2f}s · {len(result.get('sources', []))} sources retrieved")

                    # Citations
                    if result.get("sources"):
                        st.markdown("#### 📎 Citations")
                        for src in result["sources"]:
                            src_type = src.get("type", "pdf")
                            css_class = f"{src_type}-source"
                            tag_label = src_type.upper()
                            st.markdown(f"""
                            <div class="citation-card {css_class}">
                                <span class="source-tag">{tag_label}</span>
                                <strong>{src.get('source', 'Unknown')}</strong>
                                {'— p.' + str(src['page']) if src.get('page') else ''}
                                <br><span style="color:var(--muted)">{src.get('excerpt', '')[:200]}...</span>
                            </div>
                            """, unsafe_allow_html=True)

                    # Context chunks
                    if show_ctx and result.get("context_chunks"):
                        with st.expander("📦 Retrieved context chunks"):
                            for i, chunk in enumerate(result["context_chunks"]):
                                st.markdown(f"**Chunk {i+1}** — Score: `{chunk.get('score', 0):.3f}`")
                                st.code(chunk.get("text", "")[:500], language=None)
                                st.divider()

                except Exception as e:
                    st.error(f"Query failed: {e}")

# ── TAB 3: HISTORY ───────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### Query History")
    if not st.session_state.messages:
        st.info("No queries yet. Start asking questions in the **Query** tab.")
    else:
        for i, msg in enumerate(reversed(st.session_state.messages)):
            with st.expander(f"Q{len(st.session_state.messages)-i}: {msg['query'][:80]}"):
                st.markdown(f'<div class="chat-message user-msg">❓ {msg["query"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message bot-msg">{msg["answer"]}</div>', unsafe_allow_html=True)
                if msg.get("sources"):
                    st.caption(f"Sources: {', '.join(s.get('source','?') for s in msg['sources'][:3])}")
                st.caption(f"⏱ {msg.get('latency', 0):.2f}s")
