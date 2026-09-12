"""Retrieval-Augmented Generation over a small markdown knowledge base.

Default backend is TF-IDF (zero downloads, runs anywhere). Set RAG_BACKEND=fastembed
to use MiniLM vector embeddings instead (see requirements-embeddings.txt); the code
falls back to TF-IDF automatically if the embedding model is unavailable.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.config import KNOWLEDGE_DIR, RAG_BACKEND, RAG_MIN_SCORE


def load_knowledge(directory: str = KNOWLEDGE_DIR) -> list[dict]:
    """Split every .md file into chunks (one per section heading)."""
    chunks: list[dict] = []
    base = Path(directory)
    if not base.exists():
        return chunks
    for md in sorted(base.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        parts = re.split(r"\n(?=#{1,6}\s)", text)
        for i, part in enumerate(parts):
            body = part.strip()
            if len(body) > 20:
                chunks.append({"id": f"{md.stem}#{i}", "source": md.name, "text": body})
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict], backend: str = RAG_BACKEND):
        self.chunks = chunks
        self.backend = backend
        self._vectors = None
        self._matrix = None
        self._vectorizer = None
        if chunks:
            self._fit()

    def _fit(self) -> None:
        texts = [c["text"] for c in self.chunks]
        if self.backend == "fastembed":
            try:
                from fastembed import TextEmbedding  # optional heavy dep
                import numpy as np
                self._embed = TextEmbedding()
                self._matrix = np.array(list(self._embed.embed(texts)))
                self._np = np
                return
            except Exception:
                self.backend = "tfidf"  # graceful fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        if self.backend == "fastembed":
            q = self._np.array(list(self._embed.embed([query])))[0]
            mat = self._matrix
            sims = mat @ q / (
                (self._np.linalg.norm(mat, axis=1) * self._np.linalg.norm(q)) + 1e-9)
            order = sims.argsort()[::-1][:k]
            scored = [(self.chunks[i], float(sims[i])) for i in order]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            qv = self._vectorizer.transform([query])
            sims = cosine_similarity(qv, self._matrix)[0]
            order = sims.argsort()[::-1][:k]
            scored = [(self.chunks[i], float(sims[i])) for i in order]
        return [{**c, "score": round(s, 3)} for c, s in scored if s >= RAG_MIN_SCORE]
