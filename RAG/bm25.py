"""
bm25.py
-------
Translates 004_bm25.ipynb to Ollama — pure Python, no changes needed at all.
BM25 is classic keyword/lexical search, no embeddings or LLM involved.

Why we need this alongside semantic search:
  Semantic search (VectorIndex) understands MEANING but can miss exact
  rare terms like "INC-2023-Q4-011" because the embedding model doesn't
  treat it as special — it just blends into the overall meaning of the text.

  BM25 understands FREQUENCY — rare terms across the corpus get weighted
  higher, so a specific incident ID standing out in only 2 of 10 sections
  will correctly surface those 2 sections first.
"""

import math
import re
from collections import Counter
from typing import Callable, Optional, Any, List, Dict, Tuple


class BM25Index:
    """
    Classic lexical (keyword) search using the BM25 ranking algorithm.
    Same public API as VectorIndex (add_document, search) so they can be
    combined inside a Retriever later.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: Optional[Callable] = None):
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int]             = []
        self._doc_freqs: Dict[str, int]      = {}
        self._avg_doc_len: float             = 0.0
        self._idf: Dict[str, float]          = {}
        self._index_built: bool              = False

        self.k1 = k1   # term frequency saturation — higher = TF matters more
        self.b  = b    # length normalization — higher = penalize long docs more
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        """Lowercase + split on non-word characters. Simple but effective."""
        text   = text.lower()
        tokens = re.split(r"\W+", text)
        return [t for t in tokens if t]

    # =========================================================================
    # INSERT
    # =========================================================================

    def add_document(self, document: Dict[str, Any]):
        """INSERT a single record — tokenizes and updates term-frequency stats."""
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        doc_tokens = self._tokenizer(content)
        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._update_stats_add(doc_tokens)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """BULK INSERT — add many records at once."""
        for doc in documents:
            self.add_document(doc)

    def _update_stats_add(self, doc_tokens: List[str]):
        self._doc_len.append(len(doc_tokens))
        seen = set()
        for token in doc_tokens:
            if token not in seen:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen.add(token)
        self._index_built = False   # stats changed, must rebuild before next search

    def _calculate_idf(self):
        """
        Inverse Document Frequency — rare terms get a HIGH score,
        common terms (like 'the', 'a') get a LOW score.
        """
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            self._idf[term] = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)

    def _build_index(self):
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
        else:
            self._avg_doc_len = sum(self._doc_len) / len(self.documents)
            self._calculate_idf()
        self._index_built = True

    # =========================================================================
    # SELECT (search)
    # =========================================================================

    def _compute_bm25_score(self, query_tokens: List[str], doc_index: int) -> float:
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length      = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue
            idf       = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)
            numerator   = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self._avg_doc_len))
            score += numerator / (denominator + 1e-9)

        return score

    def search(self, query_text: str, k: int = 1, score_normalization_factor: float = 0.1) -> List[Tuple[Dict, float]]:
        """SELECT the k most relevant records using keyword matching + term rarity weighting."""
        if not self.documents:
            return []
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()
        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query_text)
        if not query_tokens:
            return []

        raw_scores = []
        for i in range(len(self.documents)):
            raw_score = self._compute_bm25_score(query_tokens, i)
            if raw_score > 1e-9:
                raw_scores.append((raw_score, self.documents[i]))

        raw_scores.sort(key=lambda item: item[0], reverse=True)

        # Normalize so lower = more similar (matches VectorIndex's "distance" convention)
        normalized = []
        for raw_score, doc in raw_scores[:k]:
            normalized.append((doc, math.exp(-score_normalization_factor * raw_score)))
        normalized.sort(key=lambda item: item[1])

        return normalized

    # =========================================================================
    # DELETE (not in original notebook, added for CRUD completeness)
    # =========================================================================

    def delete_document(self, index: int):
        """DELETE a record and rebuild stats from scratch (BM25 stats are global,
        so partial removal requires a full rebuild)."""
        if index < 0 or index >= len(self.documents):
            raise IndexError(f"No record at index {index}.")
        del self.documents[index]
        del self._corpus_tokens[index]
        del self._doc_len[index]

        # Rebuild doc_freqs from scratch — cheaper than tracking decrements correctly
        self._doc_freqs = {}
        for tokens in self._corpus_tokens:
            for token in set(tokens):
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
        self._index_built = False

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"BM25Index(count={len(self)}, k1={self.k1}, b={self.b}, index_built={self._index_built})"


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from chunking import chunk_by_section

    with open("report.md") as f:
        text = f.read()
    chunks = chunk_by_section(text)

    store = BM25Index()
    store.add_documents([{"content": c} for c in chunks])
    print(store)

    print("\nSearch: 'what happened with incident 2023 q4 011'")
    results = store.search("what happened with incident 2023 q4 011", k=3)
    for doc, score in results:
        print(f"  score={score:.4f} | {doc['content'][:100]}")
