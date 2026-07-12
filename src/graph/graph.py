"""
graph.py
Agentic LangGraph orchestration for the Oral Lesion Differential Reference System.

Graph structure:
    classify → route → retrieve → generate → confidence_check
                                                    ↓ low confidence (max 1 retry)
                                                clarify → retrieve → generate → END
                                                    ↓ high confidence
                                                   END

Nodes:
    classify        — classifies query into lesion category (white/red/ulcerative/mixed/vague)
                      and extracts key clinical signals (site, removability, duration, etc.)
                      Uses the LLM — first real agentic decision in the graph.

    route           — decides retrieval modality based on what inputs are available
                      (text_only | image_only | text_and_image). Pure logic, no LLM.

    retrieve        — calls appropriate LangChain tools based on route + classification.
                      Classification filters which pairs are most relevant to retrieve.

    generate        — calls local Ollama vision-LLM with retrieved context.

    confidence_check — scores the generated output for specificity.
                       If output is too vague (no pair named, no clinical test mentioned),
                       routes back for one clarification loop.
                       If specific enough, routes to END.

    clarify         — generates a targeted clarifying question based on what's missing,
                      appends it + a simulated "I'll proceed with available info" response
                      to the context, then re-routes to retrieve with refined query.
                      (In a real deployment this would pause for user input.)
"""

import os
import base64
import requests
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from src.retriever.retriever import (
    retrieve_text_chunks,
    retrieve_similar_images_by_text,
    retrieve_similar_images_by_image,
    text_retriever_tool,
    image_retriever_by_text_tool,
    image_retriever_by_image_tool,
)

OLLAMA_MODEL     = "llava:7b"
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_TEXT_MODEL = "llama3.1:8b"   # lighter model for classification + confidence check
MAX_RETRIES      = 1                 # max clarification loops before forcing END


# ── State schema ──────────────────────────────────────────────────────────────

class OralLesionState(TypedDict):
    # ── Inputs ──
    user_description:      Optional[str]
    uploaded_image_path:   Optional[str]

    # ── Set by classify node ──
    lesion_category:       Optional[str]   # white | red | ulcerative | mixed | vague
    clinical_signals:      Optional[dict]  # {site, removable, duration, systemic_signs, ...}
    refined_query:         Optional[str]   # enriched query for retrieval

    # ── Set by route node ──
    route_decision:        Optional[str]   # text_only | image_only | text_and_image

    # ── Set by retrieve node ──
    text_chunks_raw:       Optional[str]
    image_results_raw:     Optional[str]

    # ── Set by generate node ──
    generated_output:      Optional[str]

    # ── Set by confidence_check node ──
    confidence_score:      Optional[float]  # 0.0 – 1.0
    confidence_flags:      Optional[dict]   # what's missing
    retry_count:           Optional[int]

    # ── Set by clarify node ──
    clarifying_question:   Optional[str]
    clarification_context: Optional[str]   # extra context appended to next retrieval


# ── Node 1: classify ──────────────────────────────────────────────────────────

def classify_node(state: OralLesionState) -> OralLesionState:
    """
    Uses the LLM to:
    1. Categorize the lesion type (white / red / ulcerative / mixed / vague)
    2. Extract structured clinical signals from the description
    3. Build a refined query that's more specific than raw user input

   the LLM reads the description
    and decides what kind of retrieval target it is.
    """
    description = state.get("user_description", "").strip()

    if not description:
        print("[classify] No description — defaulting to vague")
        return {
            "lesion_category":  "vague",
            "clinical_signals": {},
            "refined_query":    "oral lesion differential diagnosis"
        }

    prompt = f"""You are a clinical oral pathology classifier. 
Analyze this lesion description and extract structured information.

DESCRIPTION: "{description}"

Respond in this EXACT format (no extra text):
CATEGORY: [white|red|ulcerative|mixed|vague]
SITE: [buccal mucosa|tongue|palate|gingiva|floor of mouth|lip|unknown]
REMOVABLE: [yes|no|unknown]
DURATION: [acute|chronic|unknown]
SYSTEMIC_SIGNS: [yes|no|unknown]
BILATERAL: [yes|no|unknown]
KEY_FEATURE: [one key clinical finding or unknown]
REFINED_QUERY: [a 10-20 word clinical query optimized for retrieval]"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_TEXT_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        text = response.json().get("response", "")
    except Exception as e:
        print(f"[classify] LLM call failed: {e} — using defaults")
        return {
            "lesion_category":  "vague",
            "clinical_signals": {},
            "refined_query":    description
        }

    # Parse structured response
    lines = {line.split(":")[0].strip(): line.split(":", 1)[1].strip()
             for line in text.strip().split("\n") if ":" in line}

    category = lines.get("CATEGORY", "vague").lower()
    if category not in ("white", "red", "ulcerative", "mixed", "vague"):
        category = "vague"

    signals = {
        "site":           lines.get("SITE", "unknown"),
        "removable":      lines.get("REMOVABLE", "unknown"),
        "duration":       lines.get("DURATION", "unknown"),
        "systemic_signs": lines.get("SYSTEMIC_SIGNS", "unknown"),
        "bilateral":      lines.get("BILATERAL", "unknown"),
        "key_feature":    lines.get("KEY_FEATURE", "unknown"),
    }

    refined_query = lines.get("REFINED_QUERY", description)

    print(f"[classify] category={category} | site={signals['site']} | "
          f"removable={signals['removable']} | key_feature={signals['key_feature']}")

    return {
        "lesion_category":  category,
        "clinical_signals": signals,
        "refined_query":    refined_query,
    }


# ── Node 2: route ─────────────────────────────────────────────────────────────

def route_node(state: OralLesionState) -> OralLesionState:
    """decides retrieval modality based on available inputs."""
    has_text  = bool(state.get("refined_query") or state.get("user_description", "").strip())
    has_image = bool(state.get("uploaded_image_path"))

    if has_text and has_image:
        decision = "text_and_image"
    elif has_image:
        decision = "image_only"
    else:
        decision = "text_only"

    print(f"[route] {decision}")
    return {"route_decision": decision}


# ── Conditional edge: route → retrieve ───────────────────────────────────────

def route_decision(state: OralLesionState) -> str:
    return state["route_decision"]


# ── Node 3: retrieve ──────────────────────────────────────────────────────────

def retrieve_node(state: OralLesionState) -> OralLesionState:
    """
    Calls LangChain tools based on route decision.
    Uses refined_query (from classify node) instead of raw user_description,
    so retrieval is more targeted.
    Also prepends lesion category and clinical signals to the query
    for better semantic match.
    """
    decision    = state["route_decision"]
    category    = state.get("lesion_category", "vague")
    signals     = state.get("clinical_signals", {})
    refined_q   = state.get("refined_query") or state.get("user_description", "")
    image_path  = state.get("uploaded_image_path")
    extra_ctx   = state.get("clarification_context", "")

    # Build enriched query — category + signals + refined query
    enriched_query = f"{category} lesion"
    if signals.get("site", "unknown") != "unknown":
        enriched_query += f" {signals['site']}"
    if signals.get("removable", "unknown") != "unknown":
        removable_str = "removable" if signals["removable"] == "yes" else "non-removable"
        enriched_query += f" {removable_str}"
    enriched_query += f" {refined_q}"
    if extra_ctx:
        enriched_query += f" {extra_ctx}"

    print(f"[retrieve] mode={decision} | query={enriched_query[:80]}...")

    text_chunks_raw  = ""
    image_results_raw = ""

    if decision == "text_only":
        text_chunks_raw   = text_retriever_tool.func(enriched_query)
        image_results_raw = image_retriever_by_text_tool.func(enriched_query)

    elif decision == "image_only":
        image_results_raw = image_retriever_by_image_tool.func(image_path)
        text_chunks_raw   = text_retriever_tool.func(enriched_query)

    elif decision == "text_and_image":
        text_chunks_raw   = text_retriever_tool.func(enriched_query)
        image_results_raw = image_retriever_by_image_tool.func(image_path)

    return {
        "text_chunks_raw":   text_chunks_raw,
        "image_results_raw": image_results_raw,
    }


# ── Node 4: generate ──────────────────────────────────────────────────────────

def generate_node(state: OralLesionState) -> OralLesionState:
    """
    Calls local Ollama vision-LLM.
    Includes classification output in the prompt so the LLM knows
    what category and signals were already extracted.
    """
    description   = state.get("user_description", "lesion shown in image")
    image_path    = state.get("uploaded_image_path")
    text_raw      = state.get("text_chunks_raw", "")
    image_raw     = state.get("image_results_raw", "")
    category      = state.get("lesion_category", "unknown")
    signals       = state.get("clinical_signals", {})
    clarification = state.get("clarification_context", "")

    signals_str = " | ".join(f"{k}: {v}" for k, v in signals.items()
                             if v and v != "unknown")

    prompt = f"""You are a clinical oral pathology reference assistant helping a junior clinician narrow down a differential diagnosis.

AUTOMATED PRE-CLASSIFICATION:
  Lesion category: {category}
  Clinical signals: {signals_str or "none extracted"}

RETRIEVED CLINICAL REFERENCE:
{text_raw[:3000]}

RETRIEVED SIMILAR REFERENCE IMAGES:
{image_raw[:1000]}

CLINICIAN'S DESCRIPTION:
{description}
{f"ADDITIONAL CONTEXT: {clarification}" if clarification else ""}

Based on the retrieved references and pre-classification above, provide:
1. The most likely confusion pair this presentation belongs to
2. Which condition is more likely and why, citing specific features from the description
3. The single most important clinical test or observation to confirm or rule out each condition
4. Clear next steps recommendation

Be concise and clinically focused. Do not diagnose — this is a reference tool to support clinical judgment.
If the description is insufficient for a confident differential, say so explicitly.

IMPORTANT: This is an educational reference tool only. Always defer to professional clinical judgment."""

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            payload["images"] = [base64.b64encode(f.read()).decode("utf-8")]

    print("[generate] calling Ollama...")
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        output = response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        output = (f"Could not connect to Ollama. "
                  f"Run `ollama serve` and `ollama pull {OLLAMA_MODEL}` first.")
    except Exception as e:
        output = f"Generation error: {e}"

    print("[generate] done")
    return {"generated_output": output}


# ── Node 5: confidence_check ──────────────────────────────────────────────────

def confidence_check_node(state: OralLesionState) -> OralLesionState:
    """
    Scores the generated output for specificity using rule-based checks.
    Does NOT call the LLM — fast heuristic scoring.

    Checks:
    - Does output name a specific confusion pair? (+0.3)
    - Does output mention a specific clinical test? (+0.3)
    - Does output give a next-steps recommendation? (+0.2)
    - Is the output longer than 100 words (not a fallback error)? (+0.2)

    Score >= 0.6 → confident → END
    Score < 0.6 and retries < MAX_RETRIES → clarify → retry
    Score < 0.6 and retries >= MAX_RETRIES → END anyway
    """
    output     = state.get("generated_output", "")
    retry_count = state.get("retry_count", 0)



    # ---------- Detect generation failures ----------
    output_lower = output.lower()

    failure_patterns = [
        "generation error",
        "404",
        "could not connect",
        "connectionerror",
        "connection error",
        "timed out",
        "timeout",
        "refused",
        "internal server error",
        "bad gateway",
        "service unavailable",
    ]
    if any(pattern in output_lower for pattern in failure_patterns):
        print("[confidence_check] Generation failed.")

        return {
            "confidence_score": 0.0,
            "confidence_flags": {
                "generation_failed": True,
                "pair_identified": False,
                "clinical_test_mentioned": False,
                "next_steps_present": False,
            },
            "retry_count": retry_count,
        }


    score = 0.0
    flags = {}

    # Check 1: pair identification
    pair_keywords = [
        "lichen planus", "lichenoid", "leukoplakia", "candidiasis",
        "pemphigoid", "lupus", "sialometaplasia", "carcinoma",
        "erythroplakia", "keratosis", "frictional"
    ]
    pairs_found = [kw for kw in pair_keywords if kw.lower() in output.lower()]
    if len(pairs_found) >= 2:
        score += 0.4
        flags["pair_identified"] = True
    else:
        flags["pair_identified"] = False

    # Check 2: clinical test mentioned
    test_keywords = [
        "wipe test", "wipe", "medication history", "ANA", "biopsy",
        "DIF", "immunofluorescence", "antifungal", "Nikolsky",
        "fluorescence staining", "drug withdrawal", "irritant"
    ]
    tests_found = [kw for kw in test_keywords if kw.lower() in output.lower()]
    if tests_found:
        score += 0.3
        flags["clinical_test_mentioned"] = True
    else:
        flags["clinical_test_mentioned"] = False

    # Check 3: next steps
    next_step_keywords = ["refer", "biopsy", "prescribe", "discontinue",
                          "follow up", "monitor", "recall", "urgent"]
    if any(kw.lower() in output.lower() for kw in next_step_keywords):
        score += 0.3
        flags["next_steps_present"] = True
    else:
        flags["next_steps_present"] = False

    print(f"[confidence_check] score={score:.2f} | flags={flags} | retries={retry_count}")
    return {
    "confidence_score": score,
    "confidence_flags": flags,
    "retry_count": retry_count,
}

    
# ── Conditional edge: confidence_check → clarify or END ──────────────────────

def confidence_routing(state: OralLesionState) -> str:
    score       = state.get("confidence_score", 0.0)
    retry_count = state.get("retry_count", 0)

    if score >= 0.6 or retry_count >= MAX_RETRIES:
        print(f"[confidence_routing] → END (score={score:.2f}, retries={retry_count})")
        return "end"
    else:
        print(f"[confidence_routing] → clarify (score={score:.2f})")
        return "clarify"


# ── Node 6: clarify ───────────────────────────────────────────────────────────

def clarify_node(state: OralLesionState) -> OralLesionState:
    """
    Generates a targeted clarifying question based on what confidence_check flagged.
    In this system we auto-answer it with "proceed with available info"
    and append the missing signal context to the next retrieval query.

    In deployment, this node would pause and wait for actual user input.
    For portfolio/demo purposes it simulates the loop.
    """
    flags       = state.get("confidence_flags", {})
    signals     = state.get("clinical_signals", {})
    description = state.get("user_description", "")
    retry_count = state.get("retry_count", 0)

    # Build targeted question based on what's missing
    questions = []

    if not flags.get("pair_identified"):
        if signals.get("removable") == "unknown":
            questions.append("Can the lesion be wiped off with gauze?")
        if signals.get("bilateral") == "unknown":
            questions.append("Is the lesion present on both sides of the mouth?")
        if signals.get("systemic_signs") == "unknown":
            questions.append("Does the patient have any systemic symptoms (joint pain, rash, fatigue)?")

    if not flags.get("clinical_test_mentioned"):
        questions.append("Has any medication been started or changed recently?")

    if not questions:
        questions.append("Can you provide more detail about the lesion's border, texture, or duration?")

    clarifying_question = " ".join(questions[:2])  # limit to 2 questions

    # Auto-context: append what we know to help retrieval even without real user answer
    extra_context = ""
    if signals.get("removable") == "unknown":
        extra_context += "removability unknown — consider both leukoplakia and candidiasis "
    if signals.get("bilateral") == "unknown":
        extra_context += "symmetry unknown — consider both OLP and lichenoid "
    if signals.get("systemic_signs") == "unknown":
        extra_context += "systemic signs unknown — consider both OLP and SLE "

    print(f"[clarify] question='{clarifying_question}' | retry→{retry_count + 1}")

    return {
        "clarifying_question":   clarifying_question,
        "clarification_context": extra_context.strip(),
        "retry_count":           retry_count + 1,
    }


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Builds and compiles the agentic LangGraph StateGraph.

    Flow:
        START
          ↓
        classify  ← LLM call: categorize lesion, extract signals, build refined query
          ↓
        route     ← logic: text_only / image_only / text_and_image
          ↓ (conditional edge)
        retrieve  ← LangChain tools: text retrieval + image retrieval
          ↓
        generate  ← Ollama LLaVA: grounded differential generation
          ↓
        confidence_check  ← heuristic scoring: is output specific enough?
          ↓ (conditional edge)
          ├── score >= 0.6 or retries >= 1 → END
          └── score < 0.6 → clarify → retrieve (loop back)
    """
    graph = StateGraph(OralLesionState)

    # Add all nodes
    graph.add_node("classify",          classify_node)
    graph.add_node("route",             route_node)
    graph.add_node("retrieve",          retrieve_node)
    graph.add_node("generate",          generate_node)
    graph.add_node("confidence_check",  confidence_check_node)
    graph.add_node("clarify",           clarify_node)

    # Entry point
    graph.set_entry_point("classify")

    # Fixed edges
    graph.add_edge("classify", "route")

    # Conditional: route → retrieve (all three paths lead to retrieve)
    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "text_only":      "retrieve",
            "image_only":     "retrieve",
            "text_and_image": "retrieve",
        }
    )

    # Fixed edges through generate → confidence_check
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "confidence_check")

    # Conditional: confidence_check → END or clarify
    graph.add_conditional_edges(
        "confidence_check",
        confidence_routing,
        {
            "end":     END,
            "clarify": "clarify",
        }
    )

    # Loop: clarify → retrieve (re-retrieval with enriched context)
    graph.add_edge("clarify", "retrieve")

    return graph.compile()


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = build_graph()
    print("Graph compiled.\n")

    print("=== Test 1: white patch, smoker ===")
    result = graph.invoke({
        "user_description":    "white patch on buccal mucosa, cannot be wiped off, patient is a heavy smoker for 20 years",
        "uploaded_image_path": None
    })
    print(f"\nCategory:    {result['lesion_category']}")
    print(f"Signals:     {result['clinical_signals']}")
    print(f"Refined Q:   {result['refined_query']}")
    print(f"Route:       {result['route_decision']}")
    print(f"Confidence:  {result['confidence_score']:.2f}")
    print(f"Retries:     {result['retry_count']}")
    print(f"\nOutput:\n{result['generated_output']}\n")

    print("=== Test 2: vague description (should trigger clarify loop) ===")
    result2 = graph.invoke({
        "user_description":    "lesion in the mouth",
        "uploaded_image_path": None
    })
    print(f"\nCategory:   {result2['lesion_category']}")
    print(f"Confidence: {result2['confidence_score']:.2f}")
    print(f"Clarifying: {result2.get('clarifying_question', 'none')}")
    print(f"Retries:    {result2['retry_count']}")