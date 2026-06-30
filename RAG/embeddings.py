"""
embeddings.py
-------------
Translates 002_embeddings.ipynb to Ollama.

Claude course used VoyageAI because Anthropic doesn't offer embeddings.
Ollama DOES offer local embedding models, so we don't need any external
paid service at all — everything runs on your machine.

Before running, pull an embedding model:
  ollama pull nomic-embed-text

nomic-embed-text produces 768-dimensional vectors and is small/fast —
a good local equivalent to voyage-3-large for learning purposes.
"""

import requests

OLLAMA_URL     = "http://localhost:11434/api/embed"
EMBED_MODEL    = "nomic-embed-text"


# =============================================================================
# EMBEDDING GENERATION
#    Mirrors the transcript's generate_embedding() function:
#    accepts either a single string or a list of strings.
# =============================================================================

def generate_embedding(chunks, model: str = EMBED_MODEL):
    """
    Generate embedding(s) for the given text.

    chunks: a single string OR a list of strings
    Returns: a single embedding (list[float]) if input was a string,
             or a list of embeddings if input was a list of strings.

    Transcript: "this function is going to take in some piece of text
                 and then return an embedding for it"
    """
    is_list = isinstance(chunks, list)
    inputs  = chunks if is_list else [chunks]

    res = requests.post(OLLAMA_URL, json={"model": model, "input": inputs})
    res.raise_for_status()
    embeddings = res.json()["embeddings"]

    return embeddings if is_list else embeddings[0]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Single string
    print("=" * 55)
    print("Single text embedding")
    print("=" * 55)
    vec = generate_embedding("I'm very happy today")
    print(f"Embedding length: {len(vec)}")
    print(f"First 5 values:   {vec[:5]}")

    # List of strings (batch)
    print("\n" + "=" * 55)
    print("Batch embeddings")
    print("=" * 55)
    vecs = generate_embedding([
        "The research team encountered a bug in their data collection.",
        "The team improved infection vectors in the deployment pipeline.",
    ])
    print(f"Number of embeddings: {len(vecs)}")
    print(f"Each embedding length: {len(vecs[0])}")
