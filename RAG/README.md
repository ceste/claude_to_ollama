# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the demos

Each module is independently runnable as a script:

```bash
python3 chunking.py       # compare all three chunking strategies on a sample text
python3 embeddings.py     # generate single and batch embeddings via Ollama
python3 vectordb.py       # CRUD walkthrough of VectorIndex
python3 bm25.py           # keyword search demo using BM25Index
python3 retriever.py      # hybrid search and LLM reranking demo
python3 contextual.py     # contextual retrieval demo (LLM-generated chunk context)
python3 rag_pipeline.py   # full end-to-end RAG across all three chunkers
```

## Prerequisites

Ollama must be running locally on `http://localhost:11434`. Two models are required:

```bash
ollama pull nomic-embed-text   # embeddings (768-dim vectors)
ollama pull qwen2.5:7b         # chat/generation and reranking
```

`report.md` must exist in the project root — it's the source document for all demos. The pipeline uses it to answer questions about a fictional company annual report.

## Architecture

This project implements a RAG pipeline from scratch in pure Python (no LangChain, no vector DB libraries). Each file corresponds to one notebook from a course, translated from Anthropic/VoyageAI to Ollama.

**Data flow through the pipeline:**

```
report.md
  → chunking.py        (split text into chunks)
  → embeddings.py      (embed chunks via nomic-embed-text)
  → vectordb.py        (store vectors; cosine similarity search)
  → bm25.py            (parallel keyword index; BM25 scoring)
  → retriever.py       (merge both indexes via RRF; optional LLM rerank)
  → contextual.py      (prepend LLM-generated context blurb to each chunk)
  → rag_pipeline.py    (orchestrates all steps; sends final answer via qwen2.5:7b)
```

**Key design decisions:**

- `VectorIndex` and `BM25Index` share the same public interface (`add_document`, `add_documents`, `search`) so `Retriever` can treat them interchangeably via the `SearchIndex` Protocol.
- `Retriever` uses **Reciprocal Rank Fusion (RRF)** to merge ranked lists from multiple indexes. It over-fetches `k*5` candidates from each index before merging, so the fusion has enough material to work with.
- `embeddings.generate_embedding` accepts a `str` or `list[str]` and uses the `/api/embed` endpoint (Ollama ≥ 0.1.26) with the `input` field — not the legacy `/api/embeddings` with `prompt`.
- `contextual.py` avoids feeding the full document into every chunk's context prompt. Instead it uses the first `num_start_chunks` (document intro) plus the `num_prev_chunks` immediately preceding the current chunk.

**Chunking strategies** (in `chunking.py`):

| Strategy | Best for |
|---|---|
| `chunk_by_char` | Unstructured text; production default |
| `chunk_by_sentence` | Documents with clear sentence boundaries |
| `chunk_by_section` | Markdown with `##` headers; best quality on structured docs |
