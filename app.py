"""
app.py
Streamlit interface for the Multimodal RAG Oral Lesion Differential Reference System.
Includes patient ID input and persistent consultation history via SQLite.

Run with:
    streamlit run app.py
"""

import os
import json
import pickle
import tempfile
import faiss
import streamlit as st
from PIL import Image

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
IMAGE_META_PATH = os.path.join(BASE_DIR, "data", "index", "image_metadata.pkl")

st.set_page_config(
    page_title="Oral Lesion Differential Reference",
    page_icon="🦷",
    layout="wide"
)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_graph():
    from src.graph.graph import build_graph
    return build_graph()

@st.cache_resource
def load_db():
    from src.patient_history.patient_history import PatientHistory
    return PatientHistory()


# ── Parse image results string ────────────────────────────────────────────────

def parse_image_results(raw: str) -> list[dict]:
    results = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        entry = {}
        try:
            score_part = line.split("]")[0].replace("[Score:", "").strip()
            entry["score"] = float(score_part)
            rest = line.split("]", 1)[1].strip()
            for part in rest.split("|"):
                part = part.strip()
                if "Condition:" in part:
                    entry["condition"] = part.replace("Condition:", "").strip()
                elif "Pair:" in part:
                    entry["pair_id"] = part.replace("Pair:", "").strip()
                elif "Shows distinguishing feature:" in part:
                    entry["shows_distinguishing_feature"] = part.replace(
                        "Shows distinguishing feature:", "").strip()
                elif "Image path:" in part:
                    entry["image_path"] = part.replace("Image path:", "").strip()
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
                st.image(img_path, width="content")
            else:
                st.markdown("🖼️ *Image not yet collected*")
            st.caption(
                f"**{result.get('condition', '')}**  \n"
                f"Pair: {result.get('pair_id', '')}  \n"
                f"Score: {result.get('score', 0):.3f}"
            )


# ── Render one history record ─────────────────────────────────────────────────

def render_history_record(record: dict, db, index: int):
    """Render a single past consultation as an expander."""
    signals = record.get("clinical_signals", {})
    signals_str = ", ".join(
        f"{k}: {v}" for k, v in signals.items()
        if v and v != "unknown"
    ) or "none extracted"

    label = (
        f"🕐 {record['timestamp']}  |  "
        f"Category: {record.get('lesion_category', 'unknown')}  |  "
        f"Confidence: {record.get('confidence_score', 0):.2f}"
    )

    with st.expander(label, expanded=(index == 0)):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**Description**")
            st.write(record.get("user_description") or "*No description provided*")

            st.markdown("**Clinical signals extracted**")
            st.caption(signals_str)

            if record.get("clarifying_question"):
                st.markdown("**Clarifying question triggered**")
                st.info(record["clarifying_question"])

            st.markdown("**Generated differential**")
            st.markdown(record.get("generated_output", "*No output saved*"))

        with col2:
            st.metric("Confidence", f"{record.get('confidence_score', 0):.0%}")
            st.metric("Retries", record.get("retry_count", 0))
            st.caption(f"Route: {record.get('route_decision', '')}")
            st.caption(f"Category: {record.get('lesion_category', '')}")
            had_image = record.get("had_image", 0)
            st.caption(f"Image uploaded: {'Yes' if had_image else 'No'}")

            if st.button("🗑️ Delete", key=f"del_{record['id']}"):
                db.delete_consultation(record["id"])
                st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🦷 Oral Lesion Differential Reference")
    st.markdown(
        "A clinical reference tool for **commonly confused oral lesions**. "
        "Enter a patient ID, describe the lesion, and/or upload a photo."
    )
    st.divider()

    graph = load_graph()
    db    = load_db()

    # ── Tabs: New Consultation | Patient History ───────────────────────────────
    tab1, tab2 = st.tabs(["🔍 New Consultation", "📋 Patient History"])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1 — New Consultation
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        col_sidebar, col_main = st.columns([1, 2])

        with col_sidebar:
            st.subheader("Patient & Input")

            # Patient ID
            patient_id = st.text_input(
                "Patient ID *",
                placeholder="e.g. P001 or MRN12345",
                help="Required. Used to save and retrieve consultation history."
            )

            st.divider()

            # Input mode
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
                    st.image(
                        uploaded_file,
                        caption="Uploaded image",
                        width="content"
                    )
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(uploaded_file.getvalue())
                    tmp.flush()
                    uploaded_image_path = tmp.name

            run_button = st.button(
                "🔍 Find Differentials",
                type="primary",
                disabled=not patient_id.strip()
            )

            if not patient_id.strip():
                st.caption("⚠️ Enter a Patient ID to enable the query button.")

        with col_main:
            if run_button:
                if not user_description and not uploaded_image_path:
                    st.warning("Please enter a description or upload an image.")
                    st.stop()

                # ── Run LangGraph ─────────────────────────────────────────────
                with st.spinner("Running pipeline..."):
                    result = graph.invoke({
                        "user_description":    user_description,
                        "uploaded_image_path": uploaded_image_path,
                    })

                # ── Save to history ───────────────────────────────────────────
                result["user_description"] = user_description
                db.save_consultation(
                    patient_id=patient_id,
                    result=result,
                    had_image=bool(uploaded_image_path)
                )
                st.success(
                    f"Consultation saved for patient **{patient_id.strip().upper()}**"
                )

                # ── Route badge ───────────────────────────────────────────────
                route    = result.get("route_decision", "unknown")
                category = result.get("lesion_category", "unknown")
                conf     = result.get("confidence_score", 0)
                retries  = result.get("retry_count", 0)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Category",   category)
                m2.metric("Route",      route)
                m3.metric("Confidence", f"{conf:.0%}")
                m4.metric("Retries",    retries)

                if result.get("clarifying_question"):
                    st.info(
                        f"**Clarifying question generated:** "
                        f"{result['clarifying_question']}"
                    )

                st.divider()

                # ── Reference images ──────────────────────────────────────────
                st.subheader("📸 Retrieved Reference Images")
                image_results = parse_image_results(
                    result.get("image_results_raw", "")
                )

                if image_results:
                    distinguishing = [
                        r for r in image_results
                        if r.get("shows_distinguishing_feature") == "yes"
                    ]
                    general = [
                        r for r in image_results
                        if r.get("shows_distinguishing_feature") != "yes"
                    ]
                    if distinguishing:
                        display_image_grid(
                            distinguishing,
                            "Key diagnostic images *(show the distinguishing feature)*"
                        )
                    if general:
                        display_image_grid(general, "Additional reference images")
                else:
                    st.info("No reference images — build the image index first.")

                st.divider()

                # ── Text context ──────────────────────────────────────────────
                text_raw = result.get("text_chunks_raw", "")
                if text_raw:
                    with st.expander("📄 Retrieved clinical reference text", expanded=False):
                        st.text(text_raw)

                # ── Generated differential ────────────────────────────────────
                st.subheader("🩺 Generated Differential Reference")
                st.markdown(result.get("generated_output", "No output generated."))

                # Cleanup temp file
                if uploaded_image_path and os.path.exists(uploaded_image_path):
                    os.unlink(uploaded_image_path)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2 — Patient History
    # ════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Patient History")

        all_ids = db.get_all_patient_ids()

        if not all_ids:
            st.info("No consultations saved yet. Run a query in the New Consultation tab.")
            return

        col_select, col_actions = st.columns([2, 1])

        with col_select:
            selected_id = st.selectbox(
                "Select patient",
                options=all_ids,
                format_func=lambda x: f"Patient {x}"
            )

        with col_actions:
            st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical alignment spacer
            export_btn = st.button("📥 Export JSON")
            delete_btn = st.button("🗑️ Delete all records for this patient")

        if selected_id:
            # Stats
            stats = db.get_summary_stats(selected_id)
            s1, s2, s3 = st.columns(3)
            s1.metric("Total consultations", stats.get("total", 0))
            s2.metric(
                "Avg confidence",
                f"{(stats.get('avg_confidence') or 0):.0%}"
            )
            s3.metric(
                "Categories seen",
                stats.get("categories", "none") or "none"
            )

            st.divider()

            # Export
            if export_btn:
                json_str = db.export_patient_json(selected_id)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"patient_{selected_id}_history.json",
                    mime="application/json"
                )

            # Delete all
            if delete_btn:
                db.delete_patient(selected_id)
                st.success(f"All records for patient {selected_id} deleted.")
                st.rerun()

            # Consultation history
            records = db.get_patient_history(selected_id)
            st.markdown(f"**{len(records)} consultation(s) — newest first**")

            for i, record in enumerate(records):
                render_history_record(record, db, i)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** Educational and clinical reference tool only. "
        "Not a diagnostic device. All outputs must be interpreted by a qualified "
        "dental or medical professional. Biopsy takes precedence for any lesion "
        "with malignant potential."
    )


if __name__ == "__main__":
    main()
