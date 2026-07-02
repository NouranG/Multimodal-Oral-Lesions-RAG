"""
chunk_generator.py
Loads each pair JSON and generates flat text chunks for embedding.
Each chunk is self-contained — it includes enough context to be useful
when retrieved in isolation.
"""

import json
import os
from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str        # e.g. "OLP_vs_Lichenoid__shared_features"
    pair_id: str         # e.g. "OLP_vs_Lichenoid"
    condition_a: str
    condition_b: str
    chunk_type: str      # shared_features | distinguishing | clinical_test | appearance | sites | risk | biopsy
    text: str            # the actual text that gets embedded


def load_pair(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def chunk_pair(pair: dict) -> list[TextChunk]:
    """Flatten one pair dict into multiple TextChunk objects."""
    chunks = []
    pid = pair["pair_id"]
    ca = pair["condition_a"]
    cb = pair["condition_b"]

    def make(chunk_type: str, text: str) -> TextChunk:
        return TextChunk(
            chunk_id=f"{pid}__{chunk_type}",
            pair_id=pid,
            condition_a=ca,
            condition_b=cb,
            chunk_type=chunk_type,
            text=text.strip()
        )

    # --- Chunk 1: Overview ---
    chunks.append(make("overview",
        f"Confusion pair: {ca} vs {cb}.\n"
        f"These two conditions are commonly confused because they share similar clinical features."
    ))

    # --- Chunk 2: Shared features ---
    shared = "\n".join(f"- {f}" for f in pair.get("shared_features", []))
    chunks.append(make("shared_features",
        f"Shared features between {ca} and {cb} — why they are confused:\n{shared}"
    ))

    # --- Chunk 3: Distinguishing features ---
    df = pair.get("distinguishing_features", {})
    key = df.get("key_distinguisher", "")
    a_feats = "\n".join(f"  - {f}" for f in df.get("condition_a", {}).get("features", []))
    b_feats = "\n".join(f"  - {f}" for f in df.get("condition_b", {}).get("features", []))
    chunks.append(make("distinguishing_features",
        f"How to distinguish {ca} from {cb}.\n"
        f"Key distinguisher: {key}\n\n"
        f"{ca}:\n{a_feats}\n\n"
        f"{cb}:\n{b_feats}"
    ))

    # --- Chunk 4: Clinical test ---
    ct = pair.get("clinical_test", {})
    if ct:
        interp = ct.get("result_interpretation", {})
        interp_text = "\n".join(f"  - {v}" for v in interp.values())
        chunks.append(make("clinical_test",
            f"Clinical test to distinguish {ca} from {cb}.\n"
            f"Test: {ct.get('test', '')}\n"
            f"Approach: {ct.get('approach', '')}\n"
            f"Result interpretation:\n{interp_text}"
        ))

    # --- Chunk 5: Appearance ---
    app = pair.get("appearance", {})
    if app:
        app_text = "\n".join(f"  {k}: {v}" for k, v in app.items())
        chunks.append(make("appearance",
            f"Clinical appearance of {ca} vs {cb}:\n{app_text}"
        ))

    # --- Chunk 6: Sites ---
    sites = pair.get("sites", {})
    if sites:
        sites_text = "\n".join(f"  {k}: {v}" for k, v in sites.items())
        chunks.append(make("sites",
            f"Anatomical sites affected — {ca} vs {cb}:\n{sites_text}"
        ))

    # --- Chunk 7: Risk ---
    risk = pair.get("risk", {})
    if risk:
        risk_text = "\n".join(f"  {k}: {v}" for k, v in risk.items())
        chunks.append(make("risk",
            f"Risk profile — {ca} vs {cb}:\n{risk_text}"
        ))

    # --- Chunk 8: Biopsy note ---
    biopsy = pair.get("biopsy_note", "")
    if biopsy:
        chunks.append(make("biopsy",
            f"Biopsy and histopathology notes for {ca} vs {cb}:\n{biopsy}"
        ))

    # --- Chunk 9: Treatment (if present) ---
    tx = pair.get("treatment", {})
    if tx:
        tx_text = ""
        for cond, steps in tx.items():
            if isinstance(steps, list):
                tx_text += f"  {cond}:\n" + "\n".join(f"    - {s}" for s in steps) + "\n"
        chunks.append(make("treatment",
            f"Treatment — {ca} vs {cb}:\n{tx_text}"
        ))

    # --- Chunk 10: Danger/critical notes (if present) ---
    danger = pair.get("danger_note", "")
    if danger:
        chunks.append(make("danger_note",
            f"CRITICAL CLINICAL NOTE — {ca} vs {cb}:\n{danger}"
        ))

    return chunks


def load_all_chunks(json_dir: str) -> list[TextChunk]:
    """Load all pair JSON files and return all chunks."""
    all_chunks = []
    files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))
    for fname in files:
        pair = load_pair(os.path.join(json_dir, fname))
        chunks = chunk_pair(pair)
        all_chunks.extend(chunks)
        print(f"  {fname}: {len(chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    chunks = load_all_chunks("data/text")
    print(f"\nTotal chunks: {len(chunks)}")
    print("\nSample chunk:")
    print(f"  ID: {chunks[0].chunk_id}")
    print(f"  Type: {chunks[0].chunk_type}")
    print(f"  Text preview: {chunks[0].text[:200]}...")
