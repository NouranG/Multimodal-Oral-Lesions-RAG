# Multimodal RAG System for Oral Lesion Differential Reference

A Retrieval-Augmented Generation system combining image and text retrieval
to support differential diagnosis of commonly confused oral lesions,
structured around distinguishing features rather than general condition descriptions.

## Confusion Pairs Covered

| Pair | Key Distinguisher |
|------|-------------------|
| Oral Lichen Planus vs Lichenoid Drug Eruption | Medication history + reversibility |
| Oral Lichen Planus vs Systemic Lupus Erythematosus | ANA labwork + DIF pattern |
| Leukoplakia vs Pseudomembranous Candidiasis | Wipe test + fluorescence staining |
| Leukoplakia vs Frictional Keratosis | Irritant removal + 2-week trial |
| Erosive Lichen Planus vs Mucous Membrane Pemphigoid | DIF (fibrinogen vs linear IgG/IgA) + Nikolsky sign |
| Necrotizing Sialometaplasia vs Squamous Cell Carcinoma | Course: self-healing vs expanding |
| Erythroplakia vs Erythematous Candidiasis | Antifungal response + fluorescence staining |

## Project Structure

```
oral_lesion_rag/
├── data/
│   ├── json/           # Structured pair documents (source of truth)
│   ├── images/         # Licensed oral lesion images (one folder per condition)
│   └── index/          # FAISS indices (built at runtime, not committed)
├── src/
│   ├── chunk_generator.py      # Flattens JSON pairs into text chunks
│   ├── build_text_index.py     # Embeds chunks and builds FAISS text index
│   ├── build_image_index.py    # CLIP embeddings + FAISS image index 
│   ├── retriever.py            # LangChain tool wrappers 
│   └── graph.py                # LangGraph orchestration 
├── app.py                      # Streamlit interface 
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/NouranG/Multimodal-Oral-Lesions-RAG
cd Multimodal-Oral-Lesions-RAG
pip install -r requirements.txt
```

## Build the indices (run once)

```bash
python src/build_text_index.py   # builds data/index/text_index.faiss
python src/build_image_index.py  # builds data/index/image_index.faiss
```

## Run the app

```bash
streamlit run app.py
```

## Data Sources

All text content sourced from open-access clinical references:
- StatPearls / NCBI Bookshelf (open access)
- Neville BW et al., Oral and Maxillofacial Pathology
- Peer-reviewed case reports (PMC open access, CC-BY licensed)

All images sourced from explicitly licensed open-access repositories:
- Wikimedia Commons (CC-BY / Public Domain)
- PMC open-access case reports (CC-BY licensed figures)
- License and attribution logged in data/images/metadata.csv

## Disclaimer

This system is an **educational and clinical reference tool only**.
It is not a diagnostic device and should not be used as a substitute
for clinical examination, professional judgment, or laboratory investigation.
All differentials generated must be interpreted by a qualified dental
or medical professional.
