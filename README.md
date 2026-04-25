# 🧠 Advanced Multi-Source RAG for Enterprise Knowledge Base

A production-grade RAG (Retrieval-Augmented Generation) system with multi-source ingestion,
three retrievers, result fusion, cross-encoder re-ranking, and cited answers — deployed on Streamlit.

## Architecture

```
User Query → Streamlit UI
              ↓
    ┌─── Data Ingestion Pipeline ───┐
    │  PDF | CSV | Web | Text       │
    └───────────────────────────────┘
              ↓
    ┌─── Processing ────────────────┐
    │  Chunking → Cleaning          │
    │  Embedding (OpenAI)           │
    └───────────────────────────────┘
              ↓
    ┌─── Indexing Layer ────────────┐
    │  FAISS Vector Index           │
    │  Sentence-Window Index        │
    │  BM25 Keyword Index           │
    └───────────────────────────────┘
              ↓
    ┌─── Multi-Retriever ───────────┐
    │  Retriever 1: Vector Search   │
    │  Retriever 2: Sentence Window │
    │  Retriever 3: BM25 Keyword    │
    └───────────────────────────────┘
              ↓
    ┌─── Fusion + Re-Ranking ───────┐
    │  Reciprocal Rank Fusion (RRF) │
    │  Cross-Encoder Re-Ranking     │
    └───────────────────────────────┘
              ↓
    ┌─── LLM Generation ────────────┐
    │  GPT-4o-mini / GPT-4o         │
    │  Answer + Source Citations    │
    └───────────────────────────────┘
              ↓
         Final Response
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run locally
```bash
streamlit run app.py
```

### 3. Deploy on Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Add your `OPENAI_API_KEY` in Streamlit Secrets:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

## File Structure

```
rag_project/
├── app.py              # Streamlit UI
├── rag_engine.py       # RAG pipeline (loaders, chunker, retrievers, LLM)
├── requirements.txt    # Dependencies
└── README.md
```

## Features

| Feature | Details |
|---|---|
| **Data Sources** | PDF, CSV, Website URL (crawler), Plain text |
| **Chunking** | Sliding window + Sentence-window |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector DB** | FAISS (in-memory, no server needed) |
| **Retrievers** | Vector search · Sentence-window · BM25 keyword |
| **Fusion** | Weighted Reciprocal Rank Fusion (RRF) |
| **Re-Ranking** | LLM cross-encoder scoring |
| **LLM** | GPT-4o-mini / GPT-4o (configurable) |
| **Citations** | Source name + page + excerpt |
| **UI** | Streamlit with dark editorial theme |

## Usage

1. Enter your OpenAI API Key in the sidebar
2. Go to **Data Ingestion** tab
3. Upload PDFs, CSVs, paste a URL, or enter raw text
4. Click **Build Index**
5. Switch to **Query** tab and ask questions
6. View cited answers with source snippets

## Cost Estimate (per session)

- Embedding 50 pages of PDF: ~$0.005
- 10 queries with GPT-4o-mini: ~$0.02
- Total: **< $0.03 per session**
