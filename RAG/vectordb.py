"""
vectordb.py
-----------
Translates 003_vectordb.ipynb VectorIndex class — works UNCHANGED for Ollama
since it's pure Python with zero Anthropic/VoyageAI dependency.

Extended with explicit CREATE / INSERT / UPDATE / DELETE operations,
mapped to familiar database terminology so the CRUD pattern is obvious.

  Traditional DB        VectorIndex equivalent
  ───────────────        ──────────────────────
  CREATE TABLE       →   VectorIndex()                  (instantiate the store)
  INSERT             →   add_document() / add_vector()  (add a new record)
  SELECT              →   search()                       (similarity search = "query")
  UPDATE              →   update_document()              (re-embed + replace)
  DELETE              →   delete_document()              (remove a record)
"""

import math
from typing import Optional, Any, List, Dict, Tuple


class VectorIndex:
    """
    A minimal vector database implemented in pure Python.
    Stores (vector, document) pairs and supports similarity search.

    Think of this like a table with two columns:
      vectors[i]   = the embedding for record i      (like an indexed column)
      documents[i] = the original data for record i  (like the row's other columns)
    """

    def __init__(self, distance_metric: str = "cosine", embedding_fn=None):
        """
        CREATE — equivalent to "CREATE TABLE my_vectors (...)"

        distance_metric: "cosine" or "euclidean" — how we measure similarity
        embedding_fn:    a function that turns text into a vector
                         (so add_document() can embed automatically)
        """
        self.vectors: List[List[float]]      = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int]      = None

        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_fn    = embedding_fn

    # =========================================================================
    # INSERT — adding new records
    # =========================================================================

    def add_document(self, document: Dict[str, Any]):
        """
        INSERT a single record, auto-generating its embedding from document['content'].

        Equivalent to:
          INSERT INTO my_vectors (content) VALUES ('some text')
          -- embedding is computed automatically, like an auto-generated column

        document: a dict that MUST contain a "content" key (the text to embed).
                  Can also contain other metadata keys (e.g. "title", "doc_id").
        """
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        vector = self._embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        BULK INSERT — add many records at once.
        Equivalent to: INSERT INTO my_vectors (content) VALUES (...), (...), (...)

        Note: this embeds each document one at a time internally via add_document.
        For true batch embedding (faster, fewer API calls), embed all texts
        first with generate_embedding(list_of_texts), then call add_vector()
        in a loop — see rag_pipeline.py for that pattern.
        """
        for doc in documents:
            self.add_document(doc)

    def add_vector(self, vector: List[float], document: Dict[str, Any]):
        """
        INSERT a record when you already have the embedding computed
        (e.g. from a batch embedding call).
        Equivalent to: INSERT INTO my_vectors (embedding, content) VALUES (?, ?)
        """
        if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )

        self.vectors.append(list(vector))
        self.documents.append(document)

    # =========================================================================
    # SELECT — similarity search (the "query" in RAG)
    # =========================================================================

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """
        SELECT the k most similar records to the query.
        Equivalent to:
          SELECT * FROM my_vectors ORDER BY distance(embedding, ?) LIMIT k

        query: either a string (auto-embedded) or a pre-computed vector
        k:     how many top matches to return

        Returns: list of (document, distance) tuples, sorted closest first
        """
        if not self.vectors:
            return []

        if isinstance(query, str):
            if not self._embedding_fn:
                raise ValueError("Embedding function not provided for string query.")
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_vector = query
        else:
            raise TypeError("Query must be either a string or a list of numbers.")

        if self._vector_dim is None:
            return []
        if len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}"
            )
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        dist_func = self._cosine_distance if self._distance_metric == "cosine" else self._euclidean_distance

        distances = []
        for i, stored_vector in enumerate(self.vectors):
            distance = dist_func(query_vector, stored_vector)
            distances.append((distance, i))   # keep index for update/delete by position

        distances.sort(key=lambda item: item[0])

        return [(self.documents[i], dist) for dist, i in distances[:k]]

    # =========================================================================
    # UPDATE — re-embedding and replacing an existing record
    # =========================================================================

    def update_document(self, index: int, new_content: str, extra_fields: dict = None):
        """
        UPDATE a record in place — re-embeds the new content and replaces
        both the vector and the document at the given position.
        Equivalent to:
          UPDATE my_vectors SET content = ?, embedding = embed(?) WHERE id = ?

        index:        position of the record (returned alongside search results
                       if you track indices, or via list_documents() below)
        new_content:  the new text — will be re-embedded automatically
        extra_fields: optional dict to merge into the document (e.g. {"updated": True})
        """
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided — cannot re-embed for update.")
        if index < 0 or index >= len(self.vectors):
            raise IndexError(f"No record at index {index}.")

        new_vector = self._embedding_fn(new_content)
        if len(new_vector) != self._vector_dim:
            raise ValueError("New embedding dimension does not match existing index.")

        new_doc = {"content": new_content}
        if extra_fields:
            new_doc.update(extra_fields)

        self.vectors[index]   = new_vector
        self.documents[index] = new_doc

    # =========================================================================
    # DELETE — removing a record
    # =========================================================================

    def delete_document(self, index: int):
        """
        DELETE a record by its position.
        Equivalent to: DELETE FROM my_vectors WHERE id = ?

        index: position of the record to remove
               (use list_documents() to find the index you want to delete)
        """
        if index < 0 or index >= len(self.vectors):
            raise IndexError(f"No record at index {index}.")

        del self.vectors[index]
        del self.documents[index]

    def clear(self):
        """
        DELETE all records — equivalent to: TRUNCATE TABLE my_vectors
        """
        self.vectors    = []
        self.documents  = []
        self._vector_dim = None

    # =========================================================================
    # Helper to browse records — equivalent to: SELECT * FROM my_vectors
    # =========================================================================

    def list_documents(self) -> List[Tuple[int, Dict[str, Any]]]:
        """List all stored documents with their index — useful for update/delete."""
        return list(enumerate(self.documents))

    # =========================================================================
    # Distance functions — the math behind similarity search
    # =========================================================================

    def _euclidean_distance(self, vec1, vec2) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _dot_product(self, vec1, vec2) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return sum(p * q for p, q in zip(vec1, vec2))

    def _magnitude(self, vec) -> float:
        return math.sqrt(sum(x * x for x in vec))

    def _cosine_distance(self, vec1, vec2) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        mag1, mag2 = self._magnitude(vec1), self._magnitude(vec2)
        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        cosine_similarity = self._dot_product(vec1, vec2) / (mag1 * mag2)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        return 1.0 - cosine_similarity

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        has_embed_fn = "Yes" if self._embedding_fn else "No"
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}', has_embedding_fn='{has_embed_fn}')"


# =============================================================================
# DEMO — full CRUD walkthrough
# =============================================================================

if __name__ == "__main__":
    from embeddings import generate_embedding

    store = VectorIndex(embedding_fn=generate_embedding)

    # ── CREATE (done above — VectorIndex() instantiation) ──────────────────
    print("CREATE:", store)

    # ── INSERT ───────────────────────────────────────────────────────────
    print("\n--- INSERT ---")
    store.add_document({"content": "The research team encountered a bug in their data collection."})
    store.add_document({"content": "The engineering team improved infection vectors in the pipeline."})
    store.add_document({"content": "Quarterly revenue grew by twelve percent this year."})
    print(f"After insert: {store}")
    for i, doc in store.list_documents():
        print(f"  [{i}] {doc['content'][:60]}")

    # ── SELECT (search) ─────────────────────────────────────────────────
    print("\n--- SELECT (search) ---")
    results = store.search("what happened with engineering?", k=2)
    for doc, dist in results:
        print(f"  distance={dist:.3f} | {doc['content'][:60]}")

    # ── UPDATE ───────────────────────────────────────────────────────────
    print("\n--- UPDATE ---")
    store.update_document(2, "Quarterly revenue grew by twenty percent, exceeding targets.")
    print(f"  [2] updated to: {store.documents[2]['content']}")

    # ── DELETE ───────────────────────────────────────────────────────────
    print("\n--- DELETE ---")
    store.delete_document(0)
    print(f"After delete: {store}")
    for i, doc in store.list_documents():
        print(f"  [{i}] {doc['content'][:60]}")
