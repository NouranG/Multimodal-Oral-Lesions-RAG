"""
build_image_index.py
Loads oral lesion images, embeds them using CLIP, and stores in a FAISS index.
Metadata (condition, pair_id, source, license) is stored separately in a pickle file.

Run AFTER preprocess_images.py.

Expected folder structure:
    data/images/
        lichen_planus/
        leukoplakia/
        candidiasis/
        ...
    data/images/metadata.csv
"""

import os
from pathlib import Path
import pickle
import csv
import numpy as np
import faiss
from PIL import Image
import torch
import open_clip

# ── Paths — resolved relative to project root regardless of where script is run ──

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_CSV = PROJECT_ROOT / "data" / "images" / "metadata.csv"
IMAGE_DIR = PROJECT_ROOT 
INDEX_DIR         = PROJECT_ROOT / "data" / "index"
IMAGE_INDEX_PATH  = INDEX_DIR / "image_index.faiss"
IMAGE_META_PATH   = INDEX_DIR / "image_metadata.pkl"

# CLIP model
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def load_clip_model():
    """Load CLIP model and preprocessing transform."""
    print(f"Loading CLIP model: {MODEL_NAME} ({PRETRAINED})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on: {device}")
    return model, preprocess, device


def load_metadata(csv_path: str) -> list[dict]:
    """Load image metadata from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata CSV not found at {csv_path}.\n"
            f"Run preprocess_images.py first, or check your data/images/ folder."
        )
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} image records from metadata CSV")
    return rows


def embed_images(
    metadata_rows: list[dict],
    model,
    preprocess,
    device: str
) -> tuple[np.ndarray, list[dict]]:
    """
    Embed each image using CLIP vision encoder.
    Returns:
        embeddings : np.ndarray shape (N, 512) — normalized float32
        valid_rows : metadata rows successfully embedded
                     (rows with missing/corrupt files are skipped)
    """
    embeddings = []
    valid_rows = []
    skipped = []

    for i, row in enumerate(metadata_rows):
        # image_path in CSV is relative to project root
        img_path = IMAGE_DIR / row["image_path"] if not os.path.isabs(row["image_path"]) else Path(row["image_path"])

        if not os.path.exists(img_path):
            print(f"  [SKIP] Not found: {img_path}")
            skipped.append(img_path)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            # preprocess: resize to 224x224, normalize pixel values to CLIP's expected range
            image_tensor = preprocess(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)   # (1, 512)
                # Normalize to unit vector — inner product on unit vectors = cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            embeddings.append(image_features.cpu().numpy().astype("float32"))
            valid_rows.append(row)

            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(metadata_rows)}...")

        except Exception as e:
            print(f"  [SKIP] Error on {img_path}: {e}")
            skipped.append(img_path)
            continue

    if not embeddings:
        raise RuntimeError(
            "No images were successfully embedded. "
            "Check that images exist and preprocess_images.py has been run."
        )

    embeddings_array = np.vstack(embeddings)  # list of (1,512) → (N, 512)
    print(f"\nEmbedded: {len(valid_rows)} | Skipped: {len(skipped)}")
    print(f"Embeddings shape: {embeddings_array.shape}")
    return embeddings_array, valid_rows


def build_image_index():
    model, preprocess, device = load_clip_model()

    print("\nLoading image metadata...")
    metadata_rows = load_metadata(METADATA_CSV)

    print("\nEmbedding images...")
    embeddings, valid_rows = embed_images(metadata_rows, model, preprocess, device)

    print("\nBuilding FAISS index...")
    dim = embeddings.shape[1]   # 512 for ViT-B-32
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors")

    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"\nSaving index  → {IMAGE_INDEX_PATH}")
    faiss.write_index(index, str(IMAGE_INDEX_PATH))

    print(f"Saving metadata → {IMAGE_META_PATH}")
    with open(IMAGE_META_PATH, "wb") as f:
        pickle.dump(valid_rows, f)

    print("\nImage index built successfully.")
    return index, valid_rows, model, preprocess, device


# ── Inference helpers — called by retriever.py at query time ──────────────────

def embed_query_image(
    image_path: str,
    model,
    preprocess,
    device: str
) -> np.ndarray:
    """
    Embed a user-uploaded query image using CLIP vision encoder.
    Returns normalized float32 vector of shape (1, 512).
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")


def embed_text_query_for_image_search(
    text: str,
    model,
    device: str
) -> np.ndarray:
    """
    Embed a text description into CLIP's shared embedding space.
    Because CLIP aligns text and image vectors in the same space,
    a text query like 'white striae buccal mucosa' can retrieve
    visually matching images without needing a query image.
    Returns normalized float32 vector of shape (1, 512).
    """
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    text_tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype("float32")


def retrieve_similar_images(
    query_vec: np.ndarray,
    index: faiss.Index,
    metadata: list[dict],
    k: int = 4
) -> list[dict]:
    """
    Search image FAISS index, return top-k results with metadata attached.
    Works for both image and text query vectors (both normalized in CLIP space).
    """
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        result = metadata[idx].copy()
        result["similarity_score"] = float(score)
        results.append(result)
    return results


if __name__ == "__main__":
    index, metadata, model, preprocess, device = build_image_index()

    # Text → image retrieval test (no query image needed)
    print("\n--- Text-to-image retrieval test ---")
    test_queries = [
        "white lacy striae on buccal mucosa",
        "deep crater ulcer on palate",
        "red velvety patch on floor of mouth"
    ]
    for query in test_queries:
        query_vec = embed_text_query_for_image_search(query, model, device)
        results = retrieve_similar_images(query_vec, index, metadata, k=3)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  [{r['similarity_score']:.4f}] {r['condition']} — {r['image_path']}")
