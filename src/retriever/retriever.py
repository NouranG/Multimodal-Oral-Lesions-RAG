"""
retriever.py
Wraps the FAISS text and image retrievers as LangChain Tools.
The LangGraph orchestrator calls these tools based on query type.

Two tools:
    1. TextRetrievalTool  — sentence-transformers → text FAISS index
    2. ImageRetrievalTool — CLIP → image FAISS index (image or text query)
"""

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_INDEX_PATH    = os.path.join(BASE_DIR, "data", "index", "text_index.faiss")
TEXT_META_PATH     = os.path.join(BASE_DIR, "data", "index", "text_metadata.pkl")
IMAGE_INDEX_PATH   = os.path.join(BASE_DIR, "data", "index", "image_index.faiss")
IMAGE_META_PATH    = os.path.join(BASE_DIR, "data", "index", "image_metadata.pkl")
TEXT_MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME    = "ViT-B-32"


# ── Lazy loaders — models load once on first call ─────────────────────────────

_text_model = None
_clip_model = None
_clip_preprocess = None
_clip_device = None
_text_index = None
_text_meta = None
_image_index = None
_image_meta = None


def _load_text_retriever():
    global _text_model, _text_index, _text_meta
    if _text_model is None:
        print("Loading text retriever...")
        _text_model = SentenceTransformer(TEXT_MODEL_NAME)
        _text_index = faiss.read_index(TEXT_INDEX_PATH)
        with open(TEXT_META_PATH, "rb") as f:
            _text_meta = pickle.load(f)
    return _text_model, _text_index, _text_meta


def _load_image_retriever():
    global _clip_model, _clip_preprocess, _clip_device, _image_index, _image_meta
    if _clip_model is None:
        import torch
        import open_clip
        print("Loading CLIP image retriever...")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained="openai"
        )
        _clip_model.eval()
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = _clip_model.to(_clip_device)
        _image_index = faiss.read_index(IMAGE_INDEX_PATH)
        with open(IMAGE_META_PATH, "rb") as f:
            _image_meta = pickle.load(f)
    return _clip_model, _clip_preprocess, _clip_device, _image_index, _image_meta


# ── Core retrieval functions ──────────────────────────────────────────────────

def retrieve_text_chunks(query: str, k: int = 4) -> str:
    """
    Embed query text and retrieve top-k matching text chunks from FAISS.
    Returns a formatted string — LangChain tools must return strings.
    """
    model, index, meta = _load_text_retriever()

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        m = meta[idx]
        results.append(
            f"[Score: {score:.3f}] {m['condition_a']} vs {m['condition_b']} "
            f"({m['chunk_type']}):\n{m['text']}"
        )

    return "\n\n---\n\n".join(results)


def retrieve_similar_images_by_text(query: str, k: int = 4) -> str:
    """
    Embed a text description into CLIP space and retrieve visually similar
    reference images. Returns formatted metadata string.
    CLIP's shared text-image embedding space allows this cross-modal retrieval.
    """
    import open_clip
    import torch

    model, _, device, index, meta = _load_image_retriever()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    text_tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    query_vec = text_features.cpu().numpy().astype("float32")
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        m = meta[idx]
        results.append(
            f"[Score: {score:.3f}] Condition: {m['condition']} | "
            f"Pair: {m['pair_id']} | "
            f"Shows distinguishing feature: {m.get('shows_distinguishing_feature', 'unknown')} | "
            f"Image path: {m['image_path']} | "
            f"Source: {m.get('source_url', 'N/A')} | "
            f"License: {m.get('license', 'N/A')}"
        )

    return "\n".join(results)


def retrieve_similar_images_by_image(image_path: str, k: int = 4) -> str:
    """
    Embed an uploaded image using CLIP and retrieve visually similar
    reference images from the index.
    """
    import torch
    from PIL import Image

    model, preprocess, device, index, meta = _load_image_retriever()

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    query_vec = features.cpu().numpy().astype("float32")
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        m = meta[idx]
        results.append(
            f"[Score: {score:.3f}] Condition: {m['condition']} | "
            f"Pair: {m['pair_id']} | "
            f"Shows distinguishing feature: {m.get('shows_distinguishing_feature', 'unknown')} | "
            f"Image path: {m['image_path']}"
        )

    return "\n".join(results)


# ── LangChain Tool definitions ────────────────────────────────────────────────

TextRetrievalTool = Tool(
    name="text_retriever",
    func=retrieve_text_chunks,
    description=(
        "Use this tool when the user describes a lesion in text — symptoms, appearance, "
        "site, clinical findings, or test results. "
        "Input: a clinical description string. "
        "Output: the most relevant differential diagnosis text chunks from the corpus, "
        "including distinguishing features and clinical tests."
    )
)

ImageTextRetrievalTool = Tool(
    name="image_retriever_by_text",
    func=retrieve_similar_images_by_text,
    description=(
        "Use this tool to find reference images matching a text description of a lesion. "
        "Uses CLIP's shared embedding space to bridge text and image modalities. "
        "Input: a clinical description string. "
        "Output: paths and metadata of the most visually similar reference images."
    )
)

ImageImageRetrievalTool = Tool(
    name="image_retriever_by_image",
    func=retrieve_similar_images_by_image,
    description=(
        "Use this tool when the user uploads an image of a lesion. "
        "Embeds the image using CLIP and finds visually similar reference cases. "
        "Input: file path to the uploaded image. "
        "Output: paths and metadata of the most visually similar reference images."
    )
)


# ── Tool registry — imported by the LangGraph graph ──────────────────────────

ALL_TOOLS = [TextRetrievalTool, ImageTextRetrievalTool, ImageImageRetrievalTool]


if __name__ == "__main__":
    # Quick sanity check — text retrieval only (no images needed)
    print("Testing text retrieval tool...")
    result = retrieve_text_chunks(
        "white patch that cannot be wiped off, patient smokes heavily", k=3
    )
    print(result)
