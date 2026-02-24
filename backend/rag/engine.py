"""
RAG Engine — Local FAISS-based Retrieval Augmented Generation.

Uses TF-IDF vectorization with FAISS index for fast similarity search.
Indexes from two sources:
  1. Knowledge Base (curated medical documents — heart-related)
  2. User Medical Reports (uploaded by user, stored locally)

All data stays local. PII never leaves the device.
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
BACKEND_DIR = BASE_DIR.parent          # backend/
KB_DIR = BACKEND_DIR / "knowledge_base"
REPORTS_DIR = BASE_DIR / "user_reports"
INDEX_DIR = BASE_DIR / "faiss_index"

# Ensure directories exist
KB_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)


class RAGEngine:
    """Local RAG engine with FAISS index and TF-IDF embeddings."""

    def __init__(self):
        self.documents: list[dict] = []       # {id, topic, source, content, type}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.index: Optional[faiss.IndexFlatIP] = None  # inner-product (cosine sim on normalized vectors)
        self._is_built = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_knowledge_base(self):
        """Load curated medical knowledge from JSON files."""
        for file in KB_DIR.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for entry in entries:
                    self.documents.append({
                        "id": entry.get("id", f"kb_{len(self.documents)}"),
                        "topic": entry.get("topic", ""),
                        "source": entry.get("source", file.stem),
                        "content": entry.get("content", ""),
                        "type": "knowledge_base",
                    })
            except Exception as e:
                print(f"[RAG] Error loading {file}: {e}")

    def load_user_reports(self):
        """Load user-uploaded medical reports (stored as JSON metadata + text)."""
        reports_file = REPORTS_DIR / "reports_index.json"
        if not reports_file.exists():
            return
        try:
            with open(reports_file, "r", encoding="utf-8") as f:
                reports = json.load(f)
            for report in reports:
                self.documents.append({
                    "id": report.get("id", f"report_{len(self.documents)}"),
                    "topic": report.get("filename", "User Report"),
                    "source": "user_upload",
                    "content": report.get("extracted_text", ""),
                    "type": "user_report",
                })
        except Exception as e:
            print(f"[RAG] Error loading reports: {e}")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def build_index(self):
        """Build FAISS index from all loaded documents."""
        if not self.documents:
            self.load_knowledge_base()
            self.load_user_reports()

        if not self.documents:
            print("[RAG] No documents to index.")
            return

        # Create TF-IDF matrix
        texts = [f"{d['topic']}. {d['content']}" for d in self.documents]
        self.vectorizer = TfidfVectorizer(
            max_features=4096,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray().astype("float32")

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid div by zero
        tfidf_matrix = tfidf_matrix / norms

        # Build FAISS index
        dim = tfidf_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(tfidf_matrix)

        self._is_built = True
        print(f"[RAG] Indexed {len(self.documents)} documents.")

    def rebuild_index(self):
        """Rebuild the index after new documents are added."""
        self.documents = []
        self._is_built = False
        self.build_index()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the most relevant documents for a query.
        Returns list of {id, topic, source, content, score, type}.
        """
        if not self._is_built:
            self.build_index()

        if not self._is_built or self.vectorizer is None or self.index is None:
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Search
        scores, indices = self.index.search(query_vec, min(top_k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    def get_context_for_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context and format it for injection into LLM prompt.
        Returns a formatted string ready to be appended to the sanitized prompt.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        context_parts = []
        for i, r in enumerate(results, 1):
            source_label = "Medical Knowledge" if r["type"] == "knowledge_base" else "Patient Report"
            context_parts.append(
                f"[{source_label} {i}] {r['topic']} (Source: {r['source']}, Relevance: {r['score']:.2f})\n"
                f"{r['content'][:500]}"
            )

        return (
            "\n\n--- RETRIEVED MEDICAL CONTEXT (RAG) ---\n"
            + "\n\n".join(context_parts)
            + "\n--- END CONTEXT ---\n"
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return stats about the RAG engine."""
        kb_count = sum(1 for d in self.documents if d["type"] == "knowledge_base")
        report_count = sum(1 for d in self.documents if d["type"] == "user_report")
        return {
            "total_documents": len(self.documents),
            "knowledge_base_count": kb_count,
            "user_report_count": report_count,
            "index_built": self._is_built,
            "vector_dim": self.vectorizer.max_features if self.vectorizer else 0,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get singleton RAG engine instance (builds index on first call)."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.build_index()
    return _engine


def rebuild_rag_engine() -> RAGEngine:
    """Force rebuild of the RAG engine (call after adding reports)."""
    global _engine
    _engine = RAGEngine()
    _engine.build_index()
    return _engine
