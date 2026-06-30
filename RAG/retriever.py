"""
retriever.py
------------
Translates 005_hybrid.ipynb and 006_reranking.ipynb to Ollama.

Two techniques combined here:

1. HYBRID SEARCH (multi-index retrieval)
   Run BOTH semantic search (VectorIndex) and lexical search (BM25Index)
   in parallel, merge results using Reciprocal Rank Fusion (RRF).

   Why: semantic search misses rare specific terms (like incident IDs).
        lexical search misses paraphrased/synonym questions.
        Combining both covers each other's weaknesses.

2. RERANKING
   After hybrid search returns candidates, send them to the LLM and ask
   it to re-order them by TRUE relevance to the question. This catches
   cases where neither search method alone gets the ranking quite right.

   Why: search algorithms rank by statistical/mathematical similarity.
        The LLM can actually READ and reason about relevance — slower,
        but catches subtle cases the math-based methods miss.
"""

import json
import random
import string
import requests
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5:7b"


# =============================================================================
# COMMON INTERFACE
#    Both VectorIndex and BM25Index already implement add_document/search
#    with the same signature — that's what lets us treat them interchangeably.
# =============================================================================

class SearchIndex(Protocol):
    def add_document(self, document: Dict[str, Any]): ...
    def add_documents(self, documents: List[Dict[str, Any]]): ...
    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]: ...


# =============================================================================
# RETRIEVER — combines multiple search indexes via Reciprocal Rank Fusion
# =============================================================================

class Retriever:
    """
    Wraps multiple search indexes (e.g. BM25Index + VectorIndex) and merges
    their results using Reciprocal Rank Fusion (RRF).

    RRF formula intuition: for each document, sum 1/(k_rrf + rank) across
    every index it appeared in. A document ranked #1 in BOTH indexes scores
    much higher than one ranked #1 in only one index — this rewards
    agreement between search methods.
    """

    def __init__(self, *indexes: SearchIndex, reranker_fn: Optional[Callable] = None):
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes     = list(indexes)
        self._reranker_fn = reranker_fn

    def add_document(self, document: Dict[str, Any]):
        """INSERT — assigns a random ID, then inserts into every underlying index."""
        if "id" not in document:
            document["id"] = "".join(random.choices(string.ascii_letters + string.digits, k=4))
        for index in self._indexes:
            index.add_document(document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """BULK INSERT into every underlying index."""
        for doc in documents:
            if "id" not in doc:
                doc["id"] = "".join(random.choices(string.ascii_letters + string.digits, k=4))
        for index in self._indexes:
            index.add_documents(documents)

    def search(self, query_text: str, k: int = 1, k_rrf: int = 60) -> List[Tuple[Dict[str, Any], float]]:
        """
        SELECT — runs the query against every index, merges with RRF,
        optionally reranks the final result with the reranker_fn.
        """
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        # Over-fetch (k*5) from each index so RRF has enough candidates to merge well
        all_results = [index.search(query_text, k=k * 5) for index in self._indexes]

        # Build rank table: for each doc, what rank did it get in each index?
        doc_ranks = {}
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                doc_id = id(doc)
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = {"doc_obj": doc, "ranks": [float("inf")] * len(self._indexes)}
                doc_ranks[doc_id]["ranks"][idx] = rank + 1

        def rrf_score(ranks: List[float]) -> float:
            return sum(1.0 / (k_rrf + r) for r in ranks if r != float("inf"))

        scored = [(v["doc_obj"], rrf_score(v["ranks"])) for v in doc_ranks.values()]
        scored = [(doc, s) for doc, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        result = scored[:k]

        # ── Optional reranking step ─────────────────────────────────────────
        if self._reranker_fn is not None:
            docs_only = [doc for doc, _ in result]
            for doc in docs_only:
                if "id" not in doc:
                    doc["id"] = "".join(random.choices(string.ascii_letters + string.digits, k=4))

            doc_lookup       = {doc["id"]: doc for doc in docs_only}
            reranked_ids     = self._reranker_fn(docs_only, query_text, k)
            original_scores  = {id(doc): score for doc, score in result}

            new_result = []
            for doc_id in reranked_ids:
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    new_result.append((doc, original_scores.get(id(doc), 0.0)))
            result = new_result

        return result


# =============================================================================
# LLM CALL — Ollama instead of Claude
# =============================================================================

def chat(messages: list, model: str = MODEL) -> str:
    res = requests.post(OLLAMA_URL, json={"model": model, "messages": messages, "stream": False})
    res.raise_for_status()
    return res.json()["message"]["content"]


# =============================================================================
# RERANKER FUNCTION
#    Sends candidate documents to the LLM with their IDs (not full text twice —
#    efficient: LLM just returns a reordered list of short IDs, not full content).
# =============================================================================

def reranker_fn(docs: List[Dict], query_text: str, k: int) -> List[str]:
    """
    Ask the LLM to re-order documents by true relevance to the query.
    Returns a list of document IDs in decreasing relevance order.

    Transcript: "we ask Claude to return just the IDs — much more efficient
                 than asking it to copy out the full text of each chunk."
    """
    joined_docs = "\n".join(
        f'<document><document_id>{doc["id"]}</document_id>'
        f'<document_content>{doc["content"]}</document_content></document>'
        for doc in docs
    )

    prompt = f"""You are about to be given a set of documents, along with an id of each.
Your task is to select and sort the {k} most relevant documents to answer the user's question.

Here is the user's question:
<question>{query_text}</question>

Here are the documents to select from:
<documents>{joined_docs}</documents>

Respond with ONLY a JSON object in this exact format, nothing else:
{{"document_ids": ["id1", "id2", ...]}}  // {k} ids, most relevant first
"""

    raw = chat([{"role": "user", "content": prompt}])

    # Ollama models sometimes wrap JSON in markdown fences — strip them
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(clean)["document_ids"]
    except (json.JSONDecodeError, KeyError):
        print(f"  [reranker] Failed to parse response, falling back to original order: {raw[:200]}")
        return [doc["id"] for doc in docs[:k]]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from chunking   import chunk_by_section
    from embeddings import generate_embedding
    from vectordb   import VectorIndex
    from bm25       import BM25Index

    with open("report.md") as f:
        text = f.read()
    chunks = chunk_by_section(text)

    vector_index = VectorIndex(embedding_fn=generate_embedding)
    bm25_index   = BM25Index()

    # ── Without reranker — hybrid search only ──────────────────────────────
    print("=" * 55)
    print("HYBRID SEARCH (no reranker)")
    print("=" * 55)
    retriever = Retriever(bm25_index, vector_index)
    retriever.add_documents([{"content": c} for c in chunks])

    results = retriever.search("what did the eng team do with INC-2023-Q4-011?", k=3)
    for doc, score in results:
        print(f"  score={score:.4f} | {doc['content'][:100]}")

    # ── With reranker — hybrid + LLM reranking ─────────────────────────────
    print("\n" + "=" * 55)
    print("HYBRID SEARCH + RERANKER")
    print("=" * 55)
    vector_index2 = VectorIndex(embedding_fn=generate_embedding)
    bm25_index2   = BM25Index()
    retriever2    = Retriever(bm25_index2, vector_index2, reranker_fn=reranker_fn)
    retriever2.add_documents([{"content": c} for c in chunks])

    results = retriever2.search("what did the eng team do with INC-2023-Q4-011?", k=3)
    for doc, score in results:
        print(f"  score={score:.4f} | {doc['content'][:100]}")
