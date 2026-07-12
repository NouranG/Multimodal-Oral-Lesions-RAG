# Multimodal RAG System for Oral Lesion Differential Reference

An agentic multimodal Retrieval-Augmented Generation system that helps junior clinicians navigate differential diagnosis of commonly confused oral lesions. Built with LangGraph, LangChain, CLIP, FAISS, and a local vision-language model (LLaVA via Ollama).

---

## Clinical Focus

Rather than attempting to identify any oral lesion, this system is specifically designed around **7 commonly confused lesion pairs** — cases where two conditions look similar but require different management. The distinguishing feature for each pair is the retrieval target, not just the condition name.

| Pair | Key Distinguisher |
|------|-------------------|
| Oral Lichen Planus vs Lichenoid Drug Eruption | Medication history + reversibility |
| Oral Lichen Planus vs Systemic Lupus Erythematosus | ANA labwork + honeycomb texture |
| Leukoplakia vs Pseudomembranous Candidiasis | Wipe test + fluorescence staining |
| Leukoplakia vs Frictional Keratosis | Irritant identification + 2-week elimination trial |
| Erosive Lichen Planus vs Mucous Membrane Pemphigoid | DIF pattern + Nikolsky sign |
| Necrotizing Sialometaplasia vs Squamous Cell Carcinoma | Course: self-healing vs expanding |
| Erythroplakia vs Erythematous Candidiasis | Antifungal response |

---

## System Architecture

```
User input (Streamlit)
        ↓
   graph.invoke()                    ← LangGraph agentic pipeline
        ↓
   classify_node                     ← LLM: lesion category + clinical signal extraction
        ↓
   route_node                        ← logic: text_only / image_only / text_and_image
        ↓ (conditional edge)
   retrieve_node                     ← LangChain tools: enriched query → FAISS
        │  TextRetrievalTool          → sentence-transformers → text FAISS
        │  ImageTextRetrievalTool     → CLIP text encoder → image FAISS
        │  ImageImageRetrievalTool    → CLIP image encoder → image FAISS
        ↓
   generate_node                     ← Ollama LLaVA: grounded differential generation
        ↓
   confidence_check_node             ← heuristic scoring (no LLM call)
        ↓ (conditional edge)
        ├── score ≥ 0.6 ──────────────────────────────────────→ END
        └── score < 0.6 → clarify_node → retrieve_node (loop, max 1 retry) → END
```

### Agentic Nodes

**`classify_node`** — First real decision in the graph. Calls the LLM to categorize the lesion (white / red / ulcerative / mixed / vague) and extract structured clinical signals (site, removability, duration, bilaterality, systemic signs). Builds a refined query for retrieval.

**`route_node`** — Pure logic. Checks whether the user provided text, an image, or both, and sets the retrieval modality.

**`retrieve_node`** — Calls the appropriate LangChain tools using the enriched query (refined query + category + signals), not raw user input. Handles all three retrieval paths.

**`generate_node`** — Passes retrieved text chunks, image references, classification output, and the uploaded image (base64) to the local LLaVA model via Ollama API.

**`confidence_check_node`** — Heuristic scoring (no LLM call) checking whether the output: names a specific confusion pair (+0.3), mentions a specific clinical test (+0.3), includes a next-steps recommendation (+0.2), and is substantive length (+0.2). Score ≥ 0.6 routes to END; lower scores trigger the clarify loop.

**`clarify_node`** — Generates a targeted clarifying question based on what confidence_check flagged as missing. Appends inferred context to the next retrieval query. Max one retry loop before forcing END.

---

## Project Structure

```
Multimodal-Oral-Lesions-RAG/
│
├── app.py                          # Streamlit UI
├── requirements.txt
├── README.md
│
├── data/
│   ├── json/                       # 7 structured pair documents (source of truth)
│   │   ├── pair_01_OLP_vs_Lichenoid.json
│   │   ├── pair_02_OLP_vs_SLE.json
│   │   ├── pair_03_Leukoplakia_vs_Candidiasis.json
│   │   ├── pair_04_Leukoplakia_vs_Frictional_Keratosis.json
│   │   ├── pair_05_Erosive_Lichen_Planus_vs_MMP.json
│   │   ├── pair_06_Necrotizing_Sialometaplasia_vs_SCC.json
│   │   └── pair_07_Erythroplakia_vs_Erythematous_Candidiasis.json
│   │
│   ├── images/                     # Clinical photos — collected from Wikimedia/PMC
│   │   ├── metadata.csv            # image_path, condition, pair_id, view,
│   │   │                           # shows_distinguishing_feature, source_url,
│   │   │                           # license, attribution
│   │   ├── lichen_planus/
│   │   ├── lichenoid/
│   │   ├── leukoplakia/
│   │   ├── candidiasis/
│   │   ├── sle/
│   │   ├── frictional_keratosis/
│   │   ├── erosive_lp/
│   │   ├── mmp/
│   │   ├── necrotizing_sialometaplasia/
│   │   ├── scc/
│   │   ├── erythroplakia/
│   │   └── erythematous_candidiasis/
│   │
│   └── index/                      # Built at runtime — not committed to git
│       ├── text_index.faiss
│       ├── text_metadata.pkl
│       ├── image_index.faiss
│       └── image_metadata.pkl
│
├── src/
│   ├── chunk_generator.py          # JSON → 62 flat text chunks
│   ├── build_text_index.py         # sentence-transformers → FAISS text index
│   ├── preprocess_images.py        # resize, convert, normalize filenames
│   ├── build_image_index.py        # CLIP ViT-B-32 → FAISS image index
│   ├── retriever.py                # LangChain Tool wrappers (3 tools)
│   ├── graph.py                    # LangGraph agentic pipeline (6 nodes)
│   └── evaluate.py                 # 20 clinical test queries + scoring CSV
│
└── evaluation/                     # Created when evaluate.py runs
    ├── eval_results.csv
    └── eval_raw_outputs.json
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- GPU recommended for CLIP embedding (CPU will work, slower)

### Install

```bash
git clone https://github.com/NouranG/Multimodal-Oral-Lesions-RAG
cd Multimodal-Oral-Lesions-RAG
pip install -r requirements.txt
```

Or with uv:
```bash
uv pip install -r requirements.txt
# For CUDA-accelerated PyTorch (replace cu121 with your CUDA version):
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Pull the LLM models

```bash
ollama pull llava            # vision-language model for generation
ollama pull llama3.1:8b      # text model for classification + confidence check
```

### Collect images

Fill `data/images/metadata.csv` with licensed clinical images from:
- [Wikimedia Commons](https://commons.wikimedia.org) — search condition names, filter CC-BY/Public Domain
- [PubMed Central Open Access](https://pmc.ncbi.nlm.nih.gov) — case reports with CC-BY figures

Each row in `metadata.csv`: `image_path, condition, pair_id, image_type, view, shows_distinguishing_feature, source_url, license, attribution`

### Build indices

```bash
python -m src.preprocess_images    # clean and normalize images
python -m src.build_text_index     # embed 62 text chunks → FAISS
python -m src.build_image_index    # embed images with CLIP → FAISS
```

### Run

```bash
# Start Ollama (if not already running)
ollama serve

# Launch the app
streamlit run app.py

# Run evaluation
python -m src.evaluate
```

---

## Retrieval Design

**Text retrieval** uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim) with cosine similarity via `faiss.IndexFlatIP` on normalized vectors. Each confusion pair is chunked into up to 10 typed segments (overview, shared features, distinguishing features, clinical test, appearance, sites, risk, biopsy, treatment, danger notes) — 62 chunks total.

**Image retrieval** uses OpenAI CLIP `ViT-B-32` (512-dim). Because CLIP aligns text and image embeddings in the same vector space, both text-to-image and image-to-image retrieval are supported from a single index.

**Enriched queries** — the classify node extracts structured signals before retrieval, so the retrieve node searches with `"white lesion buccal mucosa non-removable [refined query]"` rather than raw user input.

---

## Agentic Flow Example

**Input:** `"lesion in the mouth"` (vague)

```
classify  → category=vague, signals={all unknown}, refined_query="oral lesion differential"
route     → text_only
retrieve  → pulls top-k chunks across all pairs (broad)
generate  → output is generic, doesn't name a specific pair
confidence_check → score=0.2 (no pair, no test, no next steps)
clarify   → question: "Can the lesion be wiped off? Is it bilateral?"
            context: "removability unknown — consider leukoplakia and candidiasis"
retrieve  → re-retrieves with enriched context
generate  → output now more targeted
confidence_check → score=0.8 → END
```

**Input:** `"white patch buccal mucosa cannot wipe off heavy smoker 20 years"` (specific)

```
classify  → category=white, site=buccal mucosa, removable=no, refined_query="non-removable white patch buccal mucosa smoker"
route     → text_only
retrieve  → pulls Leukoplakia vs Candidiasis + Leukoplakia vs Frictional Keratosis chunks
generate  → names leukoplakia, mentions biopsy, high risk site
confidence_check → score=1.0 → END (no clarify loop needed)
```

---

## Evaluation

Run `python -m src.evaluate` to execute 20 clinical test queries (2-3 per confusion pair) through the full pipeline. Results saved to `evaluation/eval_results.csv` with blank manual scoring columns:

| Column | Description |
|--------|-------------|
| `retrieval_relevant` | Did retrieved chunks match the correct pair? (0/1) |
| `correct_pair_identified` | Did the output name the right confusion pair? (0/1) |
| `correct_condition` | Did the output favor the correct condition? (0/1) |
| `key_distinguisher_mentioned` | Was the key clinical test mentioned? (0/1) |
| `clinically_safe` | Did the output avoid overconfident diagnosis? (0/1) |

---

## Data Sources

All text content sourced from open-access clinical references:
- StatPearls / NCBI Bookshelf (open access, no license restrictions)
- Neville BW et al., *Oral and Maxillofacial Pathology*
- Peer-reviewed case reports via PubMed Central (CC-BY)

All images sourced from explicitly licensed repositories. Source URL, license, and attribution logged in `data/images/metadata.csv` for every image.

---

## Disclaimer

This system is an **educational and clinical reference tool only**. It is not a diagnostic device and must not replace clinical examination, professional judgment, laboratory investigation, or biopsy where indicated. All outputs must be interpreted by a qualified dental or medical professional. For any lesion with malignant potential — particularly erythroplakia, leukoplakia, or any non-healing ulcer — urgent referral for biopsy takes precedence over any output from this system.
