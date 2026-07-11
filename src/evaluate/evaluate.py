"""
evaluate.py
Manual evaluation of the oral lesion RAG system.

Runs 20 clinical test queries through the full LangGraph pipeline and
saves results to evaluation/eval_results.csv for manual review.

Each query includes:
    - A realistic clinical description a junior clinician might type
    - The expected confusion pair
    - The expected more-likely condition
    - The expected key distinguisher that should appear in the output

Scoring (manual — fill in eval_results.csv after reviewing outputs):
    retrieval_relevant (0/1) — did retrieved text chunks match the correct pair?
    correct_pair_identified (0/1) — did the output name the right confusion pair?
    correct_condition (0/1) — did the output favor the correct condition?
    key_distinguisher_mentioned (0/1) — did the output mention the key test/feature?
    clinically_safe (0/1) — did the output avoid overconfident diagnosis?

Run with:
    python -m src.evaluate
"""

import os
import csv
import json
import time
from datetime import datetime
from src.graph.graph import build_graph

# ── Output paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR   = os.path.join(BASE_DIR, "evaluation")
RESULTS_CSV = os.path.join(EVAL_DIR, "eval_results.csv")
RAW_JSON    = os.path.join(EVAL_DIR, "eval_raw_outputs.json")


# ── Test queries ──────────────────────────────────────────────────────────────
# 20 queries — at least 2 per confusion pair, varying clinical presentations

TEST_QUERIES = [

    # ── Pair 1: OLP vs Lichenoid Drug Eruption ────────────────────────────────
    {
        "query_id": "Q01",
        "pair": "OLP_vs_Lichenoid",
        "description": (
            "Patient presents with bilateral white lacy striae on buccal mucosa. "
            "Lesion has been present for 6 months with no clear trigger. "
            "No medications reported."
        ),
        "expected_pair": "OLP vs Lichenoid Drug Eruption",
        "expected_condition": "Oral Lichen Planus",
        "expected_distinguisher": "medication history"
    },
    {
        "query_id": "Q02",
        "pair": "OLP_vs_Lichenoid",
        "description": (
            "White striae on left buccal mucosa only, started 3 months ago. "
            "Patient was recently started on enalapril for hypertension."
        ),
        "expected_pair": "OLP vs Lichenoid Drug Eruption",
        "expected_condition": "Lichenoid Drug Eruption",
        "expected_distinguisher": "drug withdrawal resolves lesion"
    },

    # ── Pair 2: OLP vs SLE ────────────────────────────────────────────────────
    {
        "query_id": "Q03",
        "pair": "OLP_vs_SLE",
        "description": (
            "White lesion on palate with irregular honeycomb-like texture. "
            "Patient is a 35-year-old woman complaining of joint pain and fatigue. "
            "Lesion is asymmetric."
        ),
        "expected_pair": "OLP vs SLE",
        "expected_condition": "Systemic Lupus Erythematosus",
        "expected_distinguisher": "ANA blood test"
    },
    {
        "query_id": "Q04",
        "pair": "OLP_vs_SLE",
        "description": (
            "Bilateral fine white striae on buccal mucosa, symmetric. "
            "No systemic symptoms. Patient is otherwise healthy."
        ),
        "expected_pair": "OLP vs SLE",
        "expected_condition": "Oral Lichen Planus",
        "expected_distinguisher": "bilateral symmetry, no systemic signs"
    },

    # ── Pair 3: Leukoplakia vs Pseudomembranous Candidiasis ───────────────────
    {
        "query_id": "Q05",
        "pair": "Leukoplakia_vs_Candidiasis",
        "description": (
            "Creamy white patches on the palate and tongue. "
            "Patient is on broad-spectrum antibiotics for 2 weeks. "
            "Patches appear to wipe off leaving a red base."
        ),
        "expected_pair": "Leukoplakia vs Pseudomembranous Candidiasis",
        "expected_condition": "Pseudomembranous Candidiasis",
        "expected_distinguisher": "wipe test positive"
    },
    {
        "query_id": "Q06",
        "pair": "Leukoplakia_vs_Candidiasis",
        "description": (
            "Homogeneous white patch on the left buccal mucosa. "
            "Cannot be wiped off. Patient is a smoker. "
            "No systemic illness or recent antibiotic use."
        ),
        "expected_pair": "Leukoplakia vs Pseudomembranous Candidiasis",
        "expected_condition": "Leukoplakia",
        "expected_distinguisher": "non-removable, smoking history, biopsy needed"
    },

    # ── Pair 4: Leukoplakia vs Frictional Keratosis ───────────────────────────
    {
        "query_id": "Q07",
        "pair": "Leukoplakia_vs_Frictional_Keratosis",
        "description": (
            "White line on buccal mucosa exactly along the occlusal plane. "
            "Patient admits to chronic cheek biting. "
            "Lesion location corresponds precisely to the bite line."
        ),
        "expected_pair": "Leukoplakia vs Frictional Keratosis",
        "expected_condition": "Frictional Keratosis",
        "expected_distinguisher": "irritant source identified, resolves after removal"
    },
    {
        "query_id": "Q08",
        "pair": "Leukoplakia_vs_Frictional_Keratosis",
        "description": (
            "White patch on floor of mouth. No sharp teeth or dentures nearby. "
            "No identifiable mechanical cause. Patient is a heavy drinker and smoker. "
            "Lesion persists for over 3 months."
        ),
        "expected_pair": "Leukoplakia vs Frictional Keratosis",
        "expected_condition": "Leukoplakia",
        "expected_distinguisher": "no irritant source, high risk site, biopsy mandatory"
    },

    # ── Pair 5: Erosive Lichen Planus vs MMP ─────────────────────────────────
    {
        "query_id": "Q09",
        "pair": "Erosive_LP_vs_MMP",
        "description": (
            "Painful erosions on the gingiva. White striae visible at the periphery "
            "of the erythematous areas. Patient is a 52-year-old woman."
        ),
        "expected_pair": "Erosive Lichen Planus vs Mucous Membrane Pemphigoid",
        "expected_condition": "Erosive Lichen Planus",
        "expected_distinguisher": "peripheral white striae present"
    },
    {
        "query_id": "Q10",
        "pair": "Erosive_LP_vs_MMP",
        "description": (
            "Raw gingival erosions with no peripheral white lines. "
            "Lateral pressure on adjacent mucosa causes peeling. "
            "Patient also reports eye irritation and scarring."
        ),
        "expected_pair": "Erosive Lichen Planus vs Mucous Membrane Pemphigoid",
        "expected_condition": "Mucous Membrane Pemphigoid",
        "expected_distinguisher": "positive Nikolsky sign, ocular involvement"
    },
    {
        "query_id": "Q11",
        "pair": "Erosive_LP_vs_MMP",
        "description": (
            "Desquamative gingivitis in a 60-year-old woman. "
            "Intact blister noted on attached gingiva before it ruptured. "
            "No white striae around the erosion."
        ),
        "expected_pair": "Erosive Lichen Planus vs Mucous Membrane Pemphigoid",
        "expected_condition": "Mucous Membrane Pemphigoid",
        "expected_distinguisher": "intact blister, no striae, DIF needed"
    },

    # ── Pair 6: Necrotizing Sialometaplasia vs SCC ────────────────────────────
    {
        "query_id": "Q12",
        "pair": "NS_vs_SCC",
        "description": (
            "Deep crater-like ulcer on the hard palate. "
            "Appeared suddenly 3 weeks ago. Borders are well-defined. "
            "Patient reports recent local anesthesia injection in that area."
        ),
        "expected_pair": "Necrotizing Sialometaplasia vs SCC",
        "expected_condition": "Necrotizing Sialometaplasia",
        "expected_distinguisher": "self-healing course, trauma history"
    },
    {
        "query_id": "Q13",
        "pair": "NS_vs_SCC",
        "description": (
            "Palatal ulcer with indurated rolled edges. "
            "Growing progressively over 2 months. "
            "Patient is a heavy smoker with no history of trauma or injections."
        ),
        "expected_pair": "Necrotizing Sialometaplasia vs SCC",
        "expected_condition": "Squamous Cell Carcinoma",
        "expected_distinguisher": "expanding lesion, indurated borders, biopsy urgent"
    },
    {
        "query_id": "Q14",
        "pair": "NS_vs_SCC",
        "description": (
            "Large palatal ulcer, bilateral, present for 5 weeks. "
            "Patient reports it looks smaller than when first noticed. "
            "History of bulimia."
        ),
        "expected_pair": "Necrotizing Sialometaplasia vs SCC",
        "expected_condition": "Necrotizing Sialometaplasia",
        "expected_distinguisher": "shrinking course, bilateral, bulimia as trigger"
    },

    # ── Pair 7: Erythroplakia vs Erythematous Candidiasis ────────────────────
    {
        "query_id": "Q15",
        "pair": "Erythroplakia_vs_Erythematous_Candidiasis",
        "description": (
            "Red smooth patch in the center of the dorsal tongue. "
            "Diamond-shaped depapillated area. Patient wears full dentures."
        ),
        "expected_pair": "Erythroplakia vs Erythematous Candidiasis",
        "expected_condition": "Erythematous Candidiasis",
        "expected_distinguisher": "median rhomboid glossitis pattern, antifungal trial"
    },
    {
        "query_id": "Q16",
        "pair": "Erythroplakia_vs_Erythematous_Candidiasis",
        "description": (
            "Velvety bright red patch on the floor of the mouth. "
            "Well-demarcated borders. Did not respond to 2-week antifungal course. "
            "Patient is a 60-year-old male smoker."
        ),
        "expected_pair": "Erythroplakia vs Erythematous Candidiasis",
        "expected_condition": "Erythroplakia",
        "expected_distinguisher": "persists after antifungals, high-risk site, biopsy urgent"
    },
    {
        "query_id": "Q17",
        "pair": "Erythroplakia_vs_Erythematous_Candidiasis",
        "description": (
            "Erythema precisely under the upper denture-bearing area of the palate. "
            "Patient sleeps with dentures in. Denture hygiene is poor."
        ),
        "expected_pair": "Erythroplakia vs Erythematous Candidiasis",
        "expected_condition": "Erythematous Candidiasis",
        "expected_distinguisher": "denture stomatitis pattern, antifungal trial"
    },

    # ── Cross-pair / harder queries ───────────────────────────────────────────
    {
        "query_id": "Q18",
        "pair": "Leukoplakia_vs_Candidiasis",
        "description": (
            "White patch on lateral tongue. Patient is HIV positive. "
            "Patch cannot be wiped off. No recent antibiotics."
        ),
        "expected_pair": "Leukoplakia vs Pseudomembranous Candidiasis",
        "expected_condition": "Leukoplakia",
        "expected_distinguisher": "non-removable despite immunosuppression, biopsy needed"
    },
    {
        "query_id": "Q19",
        "pair": "OLP_vs_Lichenoid",
        "description": (
            "White bilateral striae on buccal mucosa. Patient takes hydroxychloroquine "
            "for rheumatoid arthritis. Lesion appeared 4 months after starting the drug."
        ),
        "expected_pair": "OLP vs Lichenoid Drug Eruption",
        "expected_condition": "Lichenoid Drug Eruption",
        "expected_distinguisher": "hydroxychloroquine is a known culprit drug"
    },
    {
        "query_id": "Q20",
        "pair": "Erosive_LP_vs_MMP",
        "description": (
            "Gingival erosions in a 55-year-old woman. No peripheral striae. "
            "Nikolsky sign equivocal. No ocular symptoms. "
            "Difficult to tell clinically."
        ),
        "expected_pair": "Erosive Lichen Planus vs Mucous Membrane Pemphigoid",
        "expected_condition": "Uncertain — biopsy with DIF required",
        "expected_distinguisher": "DIF mandatory when clinical picture is unclear"
    },
]


# ── Run evaluation ────────────────────────────────────────────────────────────
def run_evaluation():
    os.makedirs(EVAL_DIR, exist_ok=True)
    graph = build_graph()
    print(f"Graph loaded. Running {len(TEST_QUERIES)} queries...\n")

    results = []
    raw_outputs = []

    for i, query in enumerate(TEST_QUERIES):
        print(f"[{i+1}/{len(TEST_QUERIES)}] {query['query_id']} — {query['pair']}")

        start = time.time()
        try:
            output = graph.invoke({
                "user_description":    query["description"],
                "uploaded_image_path": None   # text-only evaluation
            })
            elapsed = round(time.time() - start, 2)

            generated = output.get("generated_output", "")
            text_raw  = output.get("text_chunks_raw", "")
            img_raw   = output.get("image_results_raw", "")
            route     = output.get("route_decision", "")
            status    = "OK"

        except Exception as e:
            elapsed   = round(time.time() - start, 2)
            generated = f"ERROR: {e}"
            text_raw  = ""
            img_raw   = ""
            route     = ""
            status    = "ERROR"

        # Row for CSV — manual scoring columns left blank for human review
        row = {
            "query_id":                    query["query_id"],
            "pair":                        query["pair"],
            "expected_pair":               query["expected_pair"],
            "expected_condition":          query["expected_condition"],
            "expected_distinguisher":      query["expected_distinguisher"],
            "route_decision":              route,
            "status":                      status,
            "elapsed_sec":                 elapsed,
            # ── Manual scoring — fill these in after reviewing outputs ──
            "retrieval_relevant":          "",   # 0 or 1
            "correct_pair_identified":     "",   # 0 or 1
            "correct_condition":           "",   # 0 or 1
            "key_distinguisher_mentioned": "",   # 0 or 1
            "clinically_safe":             "",   # 0 or 1
            "notes":                       "",   # free text
        }
        results.append(row)

        # Raw output for inspection
        raw_outputs.append({
            "query_id":    query["query_id"],
            "description": query["description"],
            "generated":   generated,
            "text_chunks": text_raw,
            "image_refs":  img_raw,
        })

        print(f"  Status: {status} | Time: {elapsed}s | Route: {route}")
        print(f"  Output preview: {generated[:120]}...\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fieldnames = results[0].keys()
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved → {RESULTS_CSV}")

    # ── Save raw JSON ─────────────────────────────────────────────────────────
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, indent=2, ensure_ascii=False)
    print(f"Raw outputs saved → {RAW_JSON}")

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_count    = sum(1 for r in results if r["status"] == "OK")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    avg_time    = round(sum(float(r["elapsed_sec"]) for r in results) / len(results), 2)

    print(f"\n{'='*50}")
    print(f"Evaluation complete")
    print(f"  Total queries : {len(results)}")
    print(f"  OK            : {ok_count}")
    print(f"  Errors        : {error_count}")
    print(f"  Avg time/query: {avg_time}s")
    print(f"\nNext step: open {RESULTS_CSV} and fill in the manual scoring columns.")
    print(f"Review full outputs in {RAW_JSON}.")


if __name__ == "__main__":
    run_evaluation()