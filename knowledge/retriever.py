"""
Retrieval over the markdown notes in this folder.

Nothing fancy - these notes are short and don't change often, so a fresh
TF-IDF fit on every run is plenty fast and avoids needing to download an
embedding model (and avoids a vector DB for what's currently ~10 short
chunks). If scikit-learn isn't available for some reason, we drop to a
plain keyword-overlap scorer so retrieval still works, just less precisely.
"""

import re
from dataclasses import dataclass
from pathlib import Path

NOTES_DIR = Path(__file__).parent

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@dataclass
class Chunk:
    source: str
    heading: str
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.source}#{self.heading}"


def _load_chunks() -> list[Chunk]:
    chunks = []
    for md_file in sorted(NOTES_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        # split on "## " headings, first split is usually the "# Title" preamble
        sections = re.split(r"\n(?=## )", content)
        for section in sections:
            section = section.strip()
            if not section or not section.startswith("##"):
                continue
            heading, _, body = section.partition("\n")
            heading = heading.lstrip("# ").strip()
            chunks.append(Chunk(source=md_file.stem, heading=heading, text=body.strip()))
    return chunks


class KnowledgeRetriever:
    def __init__(self):
        self.chunks = _load_chunks()
        self._vectorizer = None
        self._matrix = None

        if _HAS_SKLEARN and self.chunks:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []

        if self._vectorizer is not None:
            return self._retrieve_tfidf(query, k)
        return self._retrieve_keyword(query, k)

    def _retrieve_tfidf(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self.chunks, similarities), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:k] if score > 0]

    def _retrieve_keyword(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        query_words = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for chunk in self.chunks:
            chunk_words = set(re.findall(r"[a-z]+", chunk.text.lower()))
            overlap = len(query_words & chunk_words)
            if overlap:
                scored.append((chunk, float(overlap)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
