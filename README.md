# LLM Experiments

Two standalone experiments for learning how LLMs work in practice — an MCP demo and a RAG pipeline — both running entirely on your local machine via Ollama (no API keys required).

---

## 1. Install Ollama

Ollama runs LLMs locally on your machine.

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download the installer from https://ollama.com/download

After installing, verify it's running:
```bash
ollama --version
```

Ollama starts a local server at `http://localhost:11434` automatically on macOS/Windows. On Linux, start it manually if needed:
```bash
ollama serve
```

---

## 2. Download the models

These two models are used across both projects:

```bash
ollama pull qwen2.5:7b        # ~4.7 GB — chat/generation (used by MCP and RAG)
ollama pull nomic-embed-text  # ~274 MB — text embeddings (used by RAG only)
ollama pull llama3.1:8b       # ~4.7 GB — chat/generation (used by Untitled.ipynb)
```

Verify the downloads:
```bash
ollama list
```

You should see both models listed.

---

## 3. Install Python dependencies

Requires Python 3.10+. Install the two third-party packages:

```bash
pip install mcp requests
```

(`pydantic` is installed automatically as a dependency of `mcp`.)

---

## 4. Run the experiments

### MCP demo (`MCP/`)

Demonstrates how an LLM can use tools via the Model Context Protocol — reading and editing documents through a local MCP server.

```bash
cd MCP
python3 main.py
```

This runs 5 demos automatically (listing tools/resources/prompts, reading a document, editing a document) then drops into an interactive chat loop where you can ask questions about the documents. Type `quit` to exit.

### RAG pipeline (`RAG/`)

Demonstrates Retrieval-Augmented Generation from scratch — no LangChain, no external vector DB. Compares three chunking strategies (character-based, sentence-based, section-based) and shows how each affects retrieval quality.

```bash
cd RAG
python3 rag_pipeline.py    # full end-to-end pipeline
```

Individual modules can also be run on their own to explore specific concepts:

```bash
python3 chunking.py        # compare the three chunking strategies
python3 embeddings.py      # generate embeddings via Ollama
python3 vectordb.py        # vector similarity search (CRUD walkthrough)
python3 bm25.py            # keyword search using BM25
python3 retriever.py       # hybrid search (vector + BM25) with optional LLM reranking
python3 contextual.py      # contextual retrieval (LLM-generated chunk context)
```

---

## Repository layout

```
MCP/
  mcp_server.py   — defines tools, resources, and prompt templates (runs as subprocess)
  mcp_client.py   — connects to the server and exposes list_tools / call_tool / etc.
  main.py         — wires client to Ollama; orchestrates the tool-call loop

RAG/
  chunking.py     — chunk_by_char / chunk_by_sentence / chunk_by_section
  embeddings.py   — generate_embedding() via nomic-embed-text
  vectordb.py     — VectorIndex: in-memory vector store with cosine/euclidean search
  bm25.py         — BM25Index: keyword search
  retriever.py    — Retriever: hybrid search via Reciprocal Rank Fusion + LLM reranking
  contextual.py   — prepend LLM-generated context blurb to each chunk before indexing
  rag_pipeline.py — orchestrates all steps end-to-end
  report.md       — source document (fictional Acme Corp annual report)
```
