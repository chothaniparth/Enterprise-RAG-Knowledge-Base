"""
rag_engine.py
─────────────────────────────────────────────────────────────────
Advanced Multi-Source RAG Engine
  · Sources  : PDF, CSV, Web, Plain Text
  · Indexing : FAISS vector index + sentence-window + BM25 keyword
  · Retrieval: 3 retrievers fused with RRF + cross-encoder re-ranking
  · LLM      : OpenAI (gpt-4o-mini / gpt-4o)
"""

from __future__ import annotations

import io
import re
import math
from typing import List, Dict, Any, Tuple, Optional

# ── stdlib / third-party ──────────────────────────────────────────────────────
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helper: lazy imports with friendly errors
# ─────────────────────────────────────────────────────────────────────────────
def _require(pkg: str, pip: str | None = None):
    import importlib, sys
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = pip or pkg
        raise ImportError(
            f"Missing package '{name}'. Run:  pip install {name}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Document model
# ─────────────────────────────────────────────────────────────────────────────
class Document:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata  # keys: source, type, page, row, url, …


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
class PDFLoader:
    @staticmethod
    def load(file_obj) -> List[Document]:
        try:
            pypdf = _require("pypdf")
            reader = pypdf.PdfReader(file_obj)
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(text, {"source": file_obj.name, "type": "pdf", "page": i + 1}))
            return docs
        except Exception:
            # fallback: PyMuPDF
            fitz = _require("fitz", "pymupdf")
            data = file_obj.read()
            pdf = fitz.open(stream=data, filetype="pdf")
            docs = []
            for i, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    docs.append(Document(text, {"source": file_obj.name, "type": "pdf", "page": i + 1}))
            return docs


class CSVLoader:
    @staticmethod
    def load(file_obj) -> List[Document]:
        pd = _require("pandas")
        df = pd.read_csv(file_obj)
        docs = []
        for i, row in df.iterrows():
            text = " | ".join(f"{col}: {val}" for col, val in row.items() if str(val) != "nan")
            docs.append(Document(text, {"source": file_obj.name, "type": "csv", "row": i + 1}))
        return docs


class WebLoader:
    @staticmethod
    def load(url: str, max_pages: int = 5) -> List[Document]:
        import requests
        from urllib.parse import urljoin, urlparse
        bs4 = _require("bs4", "beautifulsoup4")
        BeautifulSoup = bs4.BeautifulSoup

        visited, queue, docs = set(), [url], []
        base = urlparse(url).netloc

        while queue and len(visited) < max_pages:
            cur = queue.pop(0)
            if cur in visited:
                continue
            try:
                resp = requests.get(cur, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                visited.add(cur)
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)
                if text.strip():
                    docs.append(Document(text, {"source": cur, "type": "web", "url": cur}))
                # collect same-domain links
                for a in soup.find_all("a", href=True):
                    href = urljoin(cur, a["href"])
                    if urlparse(href).netloc == base and href not in visited:
                        queue.append(href)
            except Exception:
                visited.add(cur)

        return docs


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────
class Chunker:
    """Sentence-aware chunker that also creates sliding-window chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> List[Document]:
        text = doc.text
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            part = " ".join(words[i: i + self.chunk_size])
            if not part.strip():
                continue
            meta = dict(doc.metadata)
            meta["chunk_id"] = len(chunks)
            meta["excerpt"] = part[:200]
            chunks.append(Document(part, meta))
        return chunks

    def sentence_window_chunks(self, doc: Document, window: int = 3) -> List[Document]:
        """Each sentence as center + surrounding sentences as context."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", doc.text.strip())
        chunks = []
        for i, sent in enumerate(sentences):
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            context = " ".join(sentences[start:end])
            meta = dict(doc.metadata)
            meta["chunk_type"] = "sentence_window"
            meta["center_sentence"] = sent
            meta["excerpt"] = sent[:200]
            chunks.append(Document(context, meta))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedder (OpenAI)
# ─────────────────────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        BATCH = 100
        all_vecs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i: i + BATCH]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            all_vecs.extend([e.embedding for e in resp.data])
        return np.array(all_vecs, dtype="float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store (FAISS)
# ─────────────────────────────────────────────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.index = None
        self.docs: List[Document] = []

    def build(self, docs: List[Document], embedder: Embedder):
        faiss = _require("faiss")
        texts = [d.text for d in docs]
        vecs = embedder.embed_batch(texts)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        # Normalize for cosine similarity
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.docs = docs

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Document, float]]:
        faiss = _require("faiss")
        q = query_vec.reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        return [(self.docs[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i >= 0]


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Keyword Retriever
# ─────────────────────────────────────────────────────────────────────────────
class BM25Retriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.k1 = 1.5
        self.b = 0.75
        corpus = [d.text.lower().split() for d in docs]
        self._build(corpus)

    def _build(self, corpus):
        N = len(corpus)
        avg_dl = sum(len(d) for d in corpus) / max(N, 1)
        df = {}
        for doc in corpus:
            for w in set(doc):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((N - f + 0.5) / (f + 0.5) + 1) for w, f in df.items()}
        self.corpus = corpus
        self.avg_dl = avg_dl

    def search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        qwords = query.lower().split()
        scores = []
        for i, doc_words in enumerate(self.corpus):
            dl = len(doc_words)
            score = 0.0
            wc = {}
            for w in doc_words:
                wc[w] = wc.get(w, 0) + 1
            for q in qwords:
                if q in self.idf:
                    tf = wc.get(q, 0)
                    num = self.idf[q] * tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                    score += num / den
            scores.append(score)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.docs[i], scores[i]) for i in top if scores[i] > 0]


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """Fuse multiple ranked lists using weighted RRF."""
    scores: Dict[int, float] = {}
    doc_map: Dict[int, Document] = {}

    for rank_list, w in zip(ranked_lists, weights):
        for rank, (doc, _) in enumerate(rank_list):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: -scores[x])
    return [(doc_map[i], scores[i]) for i in sorted_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Encoder Re-Ranker
# ─────────────────────────────────────────────────────────────────────────────
class CrossEncoderReranker:
    """Lightweight cross-encoder using OpenAI to score relevance."""

    def __init__(self, client):
        self.client = client

    def rerank(self, query: str, candidates: List[Tuple[Document, float]], top_k: int) -> List[Tuple[Document, float]]:
        if len(candidates) <= top_k:
            return candidates

        # Score each candidate with a quick LLM relevance call (batch prompt)
        texts = [f"[{i}] {c[0].text[:300]}" for i, c in enumerate(candidates[:min(len(candidates), 20)])]
        prompt = (
            f"Query: {query}\n\n"
            "Rate each passage's relevance to the query on a scale 0-10. "
            "Reply ONLY with a JSON array of integers, one per passage, e.g. [7,3,9,...]\n\n"
            + "\n".join(texts)
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.search(r"\[.*?\]", raw, re.DOTALL)
            if raw:
                scores_list = list(map(int, re.findall(r"\d+", raw.group())))
                scored = [(candidates[i][0], float(s)) for i, s in enumerate(scores_list) if i < len(candidates)]
                scored.sort(key=lambda x: -x[1])
                return scored[:top_k]
        except Exception:
            pass

        return candidates[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        chunk_size: int = 512,
        top_k: int = 5,
        use_rerank: bool = True,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.use_rerank = use_rerank
        self.weights = weights

        self.embedder = Embedder(openai_api_key)
        self.chunker = Chunker(chunk_size=chunk_size, overlap=chunk_size // 8)

        self._raw_docs: List[Document] = []
        self._chunks: List[Document] = []
        self._sw_chunks: List[Document] = []

        self.vector_store: Optional[VectorStore] = None
        self.sw_store: Optional[VectorStore] = None
        self.bm25: Optional[BM25Retriever] = None
        self.reranker = CrossEncoderReranker(self.client) if use_rerank else None

    # ── Ingestion ────────────────────────────────────────────────────────────
    def ingest_pdfs(self, files):
        for f in files:
            self._raw_docs.extend(PDFLoader.load(f))

    def ingest_csvs(self, files):
        for f in files:
            self._raw_docs.extend(CSVLoader.load(f))

    def ingest_web(self, url: str, max_pages: int = 5):
        self._raw_docs.extend(WebLoader.load(url, max_pages))

    def ingest_text(self, text: str, source: str = "text"):
        self._raw_docs.append(Document(text, {"source": source, "type": "text"}))

    # ── Indexing ─────────────────────────────────────────────────────────────
    def build_index(self):
        if not self._raw_docs:
            raise ValueError("No documents to index.")

        # 1. Regular chunks → Vector store
        for doc in self._raw_docs:
            self._chunks.extend(self.chunker.chunk(doc))

        # 2. Sentence-window chunks → SW vector store
        for doc in self._raw_docs:
            self._sw_chunks.extend(self.chunker.sentence_window_chunks(doc, window=2))

        # 3. Build FAISS for regular chunks
        self.vector_store = VectorStore()
        self.vector_store.build(self._chunks, self.embedder)

        # 4. Build FAISS for sentence-window chunks
        self.sw_store = VectorStore()
        self.sw_store.build(self._sw_chunks, self.embedder)

        # 5. BM25 on regular chunks
        self.bm25 = BM25Retriever(self._chunks)

    # ── Query ─────────────────────────────────────────────────────────────────
    def query(self, question: str) -> Dict[str, Any]:
        if self.vector_store is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Embed query
        q_vec = self.embedder.embed_one(question)

        # Three retrievers
        vec_hits = self.vector_store.search(q_vec, self.top_k * 2)
        sw_hits  = self.sw_store.search(q_vec, self.top_k * 2)
        kw_hits  = self.bm25.search(question, self.top_k * 2)

        # Fuse
        fused = reciprocal_rank_fusion([vec_hits, sw_hits, kw_hits], weights=self.weights)

        # Re-rank
        if self.use_rerank and self.reranker:
            fused = self.reranker.rerank(question, fused, self.top_k)
        else:
            fused = fused[: self.top_k]

        # Build context
        context_chunks = [
            {"text": doc.text, "score": score, "meta": doc.metadata}
            for doc, score in fused
        ]
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['meta'].get('source','?')} | Type: {c['meta'].get('type','?')}]\n{c['text']}"
            for c in context_chunks
        )

        # Generate answer
        system_prompt = (
            "You are an expert enterprise knowledge assistant. "
            "Answer the user's question using ONLY the provided context. "
            "Always cite the source document name(s) in your answer (e.g., [Source: filename.pdf]). "
            "If the answer is not in the context, say so clearly. "
            "Be precise, well-structured, and professional."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()

        # Build source citations
        seen = set()
        sources = []
        for doc, score in fused:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "type": doc.metadata.get("type", "pdf"),
                    "page": doc.metadata.get("page"),
                    "excerpt": doc.metadata.get("excerpt", doc.text[:200]),
                    "score": round(score, 4),
                })

        return {
            "answer": answer,
            "sources": sources,
            "context_chunks": context_chunks,
        }
