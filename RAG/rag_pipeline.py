"""
rag_pipeline.py
----------------
Runs the full RAG flow with three chunking strategies so you can compare
how the choice of chunker affects retrieval quality and final answers.

CHUNKING STRATEGIES:
  char     — fixed 500-char windows with 150-char overlap (most common)
  sentence — groups of 5 sentences with 1-sentence overlap
  section  — splits on markdown "## " headers (structure-aware)

STEPS:
  PREPROCESSING (once per strategy):
    1. Chunk the document
    2. Generate embeddings (single batch call)
    3. Store (embedding, chunk) pairs in a VectorIndex

  QUERY TIME (per question × per strategy):
    4. Embed the question
    5. Retrieve top-k chunks
    6. Ask Ollama to answer using those chunks
"""

import textwrap
import requests
from chunking   import chunk_by_char, chunk_by_sentence, chunk_by_section
from embeddings import generate_embedding
from vectordb   import VectorIndex

OLLAMA_URL = "http://localhost:11434/api/chat"
CHAT_MODEL = "qwen2.5:7b"

CHUNKERS = {
    "char    ": lambda text: chunk_by_char(text, chunk_size=500, chunk_overlap=150),
    "sentence": lambda text: chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1),
    "section ": chunk_by_section,
}

QUERIES = [
    "What did the software engineering department do last year?",
    "What risk factors does this company face?",
]


# =============================================================================
# STEP 6 — send question + retrieved context to Ollama
# =============================================================================

def ask_with_context(question: str, context_chunks: list[str], model: str = CHAT_MODEL) -> str:
    context_block = "\n\n---\n\n".join(context_chunks)
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the answer isn't in the context, say so.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}"
    )
    res = requests.post(OLLAMA_URL, json={
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
    })
    res.raise_for_status()
    return res.json()["message"]["content"]


# =============================================================================
# PREPROCESSING — steps 1, 2, 3
# =============================================================================

def build_index(document_path: str, chunker) -> VectorIndex:
    with open(document_path, "r") as f:
        text = f.read()

    chunks = chunker(text)
    print(f"    [1] {len(chunks)} chunks")

    embeddings = generate_embedding(chunks)
    print(f"    [2] {len(embeddings)} embeddings generated")

    store = VectorIndex(embedding_fn=generate_embedding)
    for embedding, chunk in zip(embeddings, chunks):
        store.add_vector(vector=embedding, document={"content": chunk})
    print(f"    [3] {store}")

    return store


# =============================================================================
# QUERY TIME — steps 4, 5, 6 — runs one question against all three stores
# =============================================================================

def compare_query(stores: dict, question: str, k: int = 2):
    print(f'\n  QUESTION: "{question}"')
    print()

    for name, store in stores.items():
        print(f"  ┌─ [{name}] ───────────────────────────────────────────")

        results = store.search(question, k=k)
        for rank, (doc, dist) in enumerate(results, 1):
            snippet = doc["content"].strip().replace("\n", " ")[:100]
            print(f"  │  #{rank} dist={dist:.3f} | {snippet}...")

        context_chunks = [doc["content"] for doc, _ in results]
        answer = ask_with_context(question, context_chunks)

        for line in textwrap.wrap(answer, width=70):
            print(f"  │  {line}")

        print(f"  └{'─' * 55}")
        print()


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESSING — building one index per chunking strategy")
    print("=" * 60)

    stores = {}
    for name, chunker in CHUNKERS.items():
        print(f"\n  [{name}]")
        stores[name] = build_index("./report.md", chunker)

    for i, question in enumerate(QUERIES, 1):
        print()
        print("=" * 60)
        print(f"QUERY {i}")
        print("=" * 60)
        compare_query(stores, question)
