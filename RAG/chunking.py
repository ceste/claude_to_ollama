"""
chunking.py
-----------
Translates 001_chunking.ipynb to Ollama (no changes needed — pure Python,
no Anthropic or VoyageAI dependency in this file at all).

Three chunking strategies from the transcript:
  1. chunk_by_char     — size-based, simplest, most common in production
  2. chunk_by_sentence  — splits on sentence boundaries
  3. chunk_by_section   — structure-based, splits on markdown headers
"""

import re


# =============================================================================
# 1. SIZE-BASED CHUNKING
#    Split into fixed-length pieces with overlap to preserve context
#    at chunk boundaries.
# =============================================================================

def chunk_by_char(text: str, chunk_size: int = 500, chunk_overlap: int = 150) -> list[str]:
    """
    Split text into chunks of chunk_size characters, with chunk_overlap
    characters shared between consecutive chunks (so words/context
    at the boundary aren't lost).
    """
    chunks    = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunks.append(text[start_idx:end_idx])

        # Move start forward, but step back by chunk_overlap so
        # the next chunk repeats a bit of the previous one
        start_idx = end_idx - chunk_overlap if end_idx < len(text) else len(text)

    return chunks


# =============================================================================
# 2. SENTENCE-BASED CHUNKING
#    Groups N sentences per chunk, with sentence-level overlap.
# =============================================================================

def chunk_by_sentence(text: str, max_sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> list[str]:
    """Split text into sentences, then group them into overlapping chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks    = []
    start_idx = 0

    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))
        chunks.append(" ".join(sentences[start_idx:end_idx]))

        start_idx += max_sentences_per_chunk - overlap_sentences
        if start_idx < 0:
            start_idx = 0

    return chunks


# =============================================================================
# 3. STRUCTURE-BASED CHUNKING
#    Splits on markdown section headers ("## "). Best quality IF your
#    document has reliable structure. Falls apart on plain unstructured text.
# =============================================================================

def chunk_by_section(document_text: str) -> list[str]:
    """Split on markdown '## ' section headers."""
    pattern = r"\n## "
    return re.split(pattern, document_text)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    sample = """# Company Report

## Executive Summary
This report covers our annual performance across all departments.

## Medical Research
The research team encountered a bug in their data collection process.
They fixed it within two days and resumed normal operations.

## Software Engineering
The engineering team improved infection vectors in the deployment pipeline.
This was a major security improvement for the year.
"""

    print("=" * 55)
    print("chunk_by_char (size=150, overlap=20)")
    print("=" * 55)
    for c in chunk_by_char(sample, chunk_size=150, chunk_overlap=20):
        print(repr(c[:80]), "...\n")

    print("=" * 55)
    print("chunk_by_sentence (5 sentences, 1 overlap)")
    print("=" * 55)
    for c in chunk_by_sentence(sample):
        print(repr(c[:80]), "...\n")

    print("=" * 55)
    print("chunk_by_section")
    print("=" * 55)
    for c in chunk_by_section(sample):
        print(repr(c[:80]), "...\n")
