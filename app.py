"""
app.py
Streamlit interface for the Multimodal RAG Oral Lesion Differential Reference System.
All retrieval and generation is orchestrated through the LangGraph graph (src/graph.py).

Run with:
    streamlit run app.py
"""

import os
import pickle
import tempfile
import faiss
import streamlit as st
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
IMAGE_META_PATH  = os.path.join(BASE_DIR, "data", "index", "image_metadata.pkl")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oral Lesion Differential Reference",
    page_icon="🦷",
    layout="wide"
)


# ── Load graph (cached — compiles once per session) ───────────────────────────
@st.cache_resource
def load_graph():
    from src.graph.graph import build_graph
    return build_graph()


# ── Load image metadata for display (cached) ─────────────────────────────────
@st.cache_resource
def load_image_metadata():
    if not os.path.exists(IMAGE_META_PATH):
        return []
    with open(IMAGE_META_PATH, "rb") as f:
        return pickle.load(f)


# ── Parse image results string → list of dicts ───────────────────────────────
def parse_image_results(raw: str) -> list[dict]:
    """
    Convert the formatted string returned by the image retrieval tools
    back into a list of dicts for display in the UI.
    Each line looks like:
    [Score: 0.85] Condition: X | Pair: Y | Shows distinguishing feature: yes | Image path: Z
    """
    results = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        entry = {}
        try:
            # Extract score
            score_part = line.split("]")[0].replace("[Score:", "").strip()
            entry["score"] = float(score_part)
            # Extract fields after ]
            rest = line.split("]", 1)[1].strip()
            for part in rest.split("|"):
                part = part.strip()
                if "Condition:" in part:
                    entry["condition"] = part.replace("Condition:", "").strip()
                elif "Pair:" in part:
                    entry["pair_id"] = part.replace("Pair:", "").strip()
                elif "Shows distinguishing feature:" in part:
                    entry["shows_distinguishing_feature"] = part.replace("Shows distinguishing feature:", "").strip()
                elif "Image path:" in part:
                    entry["image_path"] = part.replace("Image path:", "").strip()
                elif "Source:" in part:
                    entry["source_url"] = part.replace("Source:", "").strip()
            results.append(entry)
        except Exception:
            continue
    return results


# ── Display image grid ────────────────────────────────────────────────────────
def display_image_grid(results: list[dict], label: str):
    if not results:
        return
    st.markdown(f"**{label}**")
    cols = st.columns(min(len(results), 4))
    for col, result in zip(cols, results):
        with col:
            img_path = result.get("image_path", "")
            if not os.path.isabs(img_path):
                img_path = os.path.join(BASE_DIR, img_path)
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
            else:
                st.markdown("🖼️ *Image not yet collected*")
            st.caption(
                f"**{result.get('condition', '')}**  \n"
                f"Pair: {result.get('pair_id', '')}  \n"
                f"Score: {result.get('score', 0):.3f}"
            )


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.title("🦷 Oral Lesion Differential Reference")
    st.markdown(
        "A clinical reference tool for **commonly confused oral lesions**. "
        "Describe the lesion and/or upload a photo — the system retrieves "
        "similar reference cases and generates a structured differential."
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Query Input")

        input_mode = st.radio(
            "Input type",
            ["Text description only", "Image only", "Text + Image"],
            index=0
        )

        user_description  = ""
        uploaded_image_path = None

        if input_mode in ["Text description only", "Text + Image"]:
            user_description = st.text_area(
                "Describe the lesion",
                placeholder=(
                    "e.g. white patch on buccal mucosa, cannot be wiped off, "
                    "patient smokes, present for 3 months"
                ),
                height=150
            )

        if input_mode in ["Image only", "Text + Image"]:
            uploaded_file = st.file_uploader(
                "Upload lesion photo",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
                suffix = os.path.splitext(uploaded_file.name)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded_file.getvalue())
                tmp.flush()
                uploaded_image_path = tmp.name

        run_button = st.button("🔍 Find Differentials", type="primary")

    # ── Main area ─────────────────────────────────────────────────────────────
    if run_button:
        if not user_description and not uploaded_image_path:
            st.warning("Please enter a description or upload an image.")
            st.stop()

        # ── Run LangGraph ─────────────────────────────────────────────────────
        with st.spinner("Running retrieval and generation pipeline..."):
            graph = load_graph()
            result = graph.invoke({
                "user_description":    user_description,
                "uploaded_image_path": uploaded_image_path,
            })

        # ── Show route decision ───────────────────────────────────────────────
        route = result.get("route_decision", "unknown")
        st.caption(f"Input mode detected: `{route}`")

        # ── Retrieved reference images ────────────────────────────────────────
        st.subheader("📸 Retrieved Reference Images")
        image_results = parse_image_results(result.get("image_results_raw", ""))

        if image_results:
            distinguishing = [r for r in image_results if r.get("shows_distinguishing_feature") == "yes"]
            general        = [r for r in image_results if r.get("shows_distinguishing_feature") != "yes"]

            if distinguishing:
                display_image_grid(
                    distinguishing,
                    "Key diagnostic images *(show the distinguishing feature)*"
                )
            if general:
                display_image_grid(general, "Additional reference images")
        else:
            st.info("No reference images retrieved — check that the image index has been built.")

        st.divider()

        # ── Retrieved text context (expandable) ───────────────────────────────
        text_raw = result.get("text_chunks_raw", "")
        if text_raw:
            with st.expander("📄 Retrieved clinical reference text", expanded=False):
                st.text(text_raw)

        # ── Generated differential ────────────────────────────────────────────
        st.subheader("🩺 Generated Differential Reference")
        generated = result.get("generated_output", "")
        if generated:
            st.markdown(generated)
        else:
            st.info("No output generated — check Ollama is running.")

        # Cleanup temp file
        if uploaded_image_path and os.path.exists(uploaded_image_path):
            os.unlink(uploaded_image_path)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This is an educational and clinical reference tool only. "
        "It is not a diagnostic device and must not replace clinical examination, "
        "professional judgment, laboratory investigation, or biopsy where indicated. "
        "All outputs must be interpreted by a qualified dental or medical professional."
    )


if __name__ == "__main__":
    main()
