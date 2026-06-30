"""
contextual.py
-------------
Translates 007_contextual.ipynb to Ollama.

The problem: when you chunk a document, each chunk loses its surrounding
context. A chunk that says "the incident was resolved within 48 hours"
doesn't say WHICH incident, or WHICH section of the report it came from.

The fix: before storing each chunk, ask the LLM to write a short blurb
situating that chunk within the larger document, then prepend that blurb
to the chunk before embedding/indexing it. This "contextualized chunk"
carries much more retrievable meaning than the raw chunk alone.

Handles the large-document problem: if the full source document is too
big to fit in a single prompt, only include:
  - the first N chunks (likely a summary/intro of the whole doc)
  - the M chunks immediately before the one being contextualized
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5:7b"


def chat(messages: list, model: str = MODEL) -> str:
    res = requests.post(OLLAMA_URL, json={"model": model, "messages": messages, "stream": False})
    res.raise_for_status()
    return res.json()["message"]["content"]


# =============================================================================
# ADD CONTEXT — the core contextual retrieval function
# =============================================================================

def add_context(text_chunk: str, source_text: str, model: str = MODEL) -> str:
    """
    Ask the LLM to write a short blurb situating this chunk within the
    larger document, then prepend it to the original chunk text.

    text_chunk:  the chunk we want to add context to
    source_text: surrounding document text used to generate the context
                 (can be the FULL document, or a reduced subset — see below)

    Returns: contextualized_chunk = "<generated context>\n<original chunk>"
    """
    prompt = f"""Write a short and succinct snippet of text to situate this chunk within the
overall source document for the purposes of improving search retrieval of the chunk.

Here is the original source document:
<document>
{source_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{text_chunk}
</chunk>

Answer only with the succinct context and nothing else."""

    context = chat([{"role": "user", "content": prompt}], model=model)
    return context.strip() + "\n" + text_chunk


# =============================================================================
# BULK CONTEXTUALIZATION
#    Handles large documents: instead of feeding the WHOLE document for
#    every single chunk (expensive, may not fit), use a reduced context:
#      - first `num_start_chunks` (the intro/summary of the doc)
#      - `num_prev_chunks` immediately preceding the current chunk
# =============================================================================

def contextualize_chunks(
    chunks:          list[str],
    num_start_chunks: int = 2,
    num_prev_chunks:  int = 2,
    model:            str = MODEL,
) -> list[str]:
    """
    Contextualize every chunk in the list using a reduced surrounding context
    instead of the full document — keeps prompts small even for huge documents.

    Returns: list of contextualized chunks, same length and order as input.
    """
    contextualized = []

    for i, chunk in enumerate(chunks):
        context_parts = []

        # Starter chunks — likely contain a summary/intro of the whole doc
        context_parts.extend(chunks[:min(num_start_chunks, len(chunks))])

        # Chunks immediately before this one — local context
        start_idx = max(0, i - num_prev_chunks)
        context_parts.extend(chunks[start_idx:i])

        context = "\n".join(context_parts)

        print(f"  Contextualizing chunk {i+1}/{len(chunks)}...")
        contextualized.append(add_context(chunk, context, model=model))

    return contextualized


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from chunking   import chunk_by_section
    from embeddings import generate_embedding
    from vectordb   import VectorIndex
    from bm25       import BM25Index
    from retriever  import Retriever, reranker_fn

    with open("report.md") as f:
        report_text = f.read()
    chunks = chunk_by_section(report_text)

    # ── Single chunk demo — see what the contextualization looks like ──────
    print("=" * 55)
    print("SINGLE CHUNK CONTEXTUALIZATION DEMO")
    print("=" * 55)
    example = add_context(chunks[2], report_text)
    print(example)

    # ── Full pipeline: contextualize all chunks, then index them ───────────
    print("\n" + "=" * 55)
    print("CONTEXTUALIZING ALL CHUNKS (reduced context for large docs)")
    print("=" * 55)
    contextualized_chunks = contextualize_chunks(chunks, num_start_chunks=2, num_prev_chunks=2)

    vector_index = VectorIndex(embedding_fn=generate_embedding)
    bm25_index   = BM25Index()
    retriever    = Retriever(bm25_index, vector_index, reranker_fn=reranker_fn)
    retriever.add_documents([{"content": c} for c in contextualized_chunks])

    print("\n" + "=" * 55)
    print("SEARCH WITH CONTEXTUALIZED CHUNKS")
    print("=" * 55)
    results = retriever.search("what did the eng team do with INC-2023-Q4-011?", k=3)
    for doc, score in results:
        print(f"  score={score:.4f} | {doc['content'][:150]}")
