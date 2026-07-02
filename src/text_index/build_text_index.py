"""
build_text_index.py
Embeds all text chunks using sentence-transformers and stores them in a FAISS index.
Also saves chunk metadata separately so we can retrieve it alongside embeddings.
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.chunk_generator.chunk_generator import load_all_chunks, TextChunk

# Paths
JSON_DIR = "data/texts"
INDEX_DIR = "data/index"
INDEX_PATH = os.path.join(INDEX_DIR, "text_index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "text_metadata.pkl")

# Model — small but strong for medical/clinical text similarity
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index():
    print("Loading chunks...")
    chunks = load_all_chunks(JSON_DIR)
    print(f"Total chunks to embed: {len(chunks)}")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks...")
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors")

    print(f"\nSaving index to {INDEX_PATH}")
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving metadata to {METADATA_PATH}")
    metadata = [
        {
            "chunk_id": c.chunk_id,
            "pair_id": c.pair_id,
            "condition_a": c.condition_a,
            "condition_b": c.condition_b,
            "chunk_type": c.chunk_type,
            "text": c.text
        }
        for c in chunks
    ]
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("\nDone. Index built successfully.")
    return index, metadata


def test_retrieval(index, metadata, query: str, k: int = 3):
    """Quick retrieval test after building the index."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)

    print(f"\nQuery: '{query}'")
    print(f"Top {k} results:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        m = metadata[idx]
        print(f"\n  [{rank+1}] Score: {score:.4f}")
        print(f"       Pair: {m['condition_a']} vs {m['condition_b']}")
        print(f"       Type: {m['chunk_type']}")
        print(f"       Preview: {m['text'][:150]}...")


if __name__ == "__main__":
    index, metadata = build_index()

    # Test with a few clinical queries
    test_retrieval(index, metadata,
        "white patch that cannot be wiped off, patient is a smoker")
    test_retrieval(index, metadata,
        "painful erosions on gingiva, blistering, positive Nikolsky sign")
    test_retrieval(index, metadata,
        "deep palatal ulcer, self healing, no growth over time")
