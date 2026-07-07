"""
graph.py
LangGraph orchestration for the Oral Lesion Differential Reference System.

Graph structure:
    route → retrieve → generate

Nodes:
    route    — inspects the input state and decides which retrieval path to take
               (text_only | image_only | text_and_image)
    retrieve — calls the appropriate LangChain tool(s) based on route decision
    generate — calls local Ollama vision-LLM with retrieved context to produce
               a grounded differential reference

State:
    A TypedDict that flows through all nodes, accumulating results at each step.

Usage:
    from src.graph import build_graph
    graph = build_graph()
    result = graph.invoke({
        "user_description": "white patch buccal mucosa cannot be wiped off",
        "uploaded_image_path": None
    })
    print(result["generated_output"])
"""

import os
import base64
import requests
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain.tools import tool

# Import our retrieval functions
from src.retriever.retriever import (
    retrieve_text_chunks,
    retrieve_similar_images_by_text,
    retrieve_similar_images_by_image,
    text_retriever_tool,
    image_retriever_by_text_tool,
    image_retriever_by_image_tool,
)

OLLAMA_MODEL = "qwen2.5vl:latest"
OLLAMA_URL   = "http://localhost:11434/api/generate"


# ── State schema ──────────────────────────────────────────────────────────────
# TypedDict defines the shape of the state dict that flows through the graph.
# Every node receives this state and returns a partial dict to update it.

class OralLesionState(TypedDict):
    # ── Inputs (set before graph.invoke()) ──
    user_description: Optional[str]        # free-text lesion description
    uploaded_image_path: Optional[str]     # path to user-uploaded image (or None)

    # ── Set by route node ──
    route_decision: Optional[str]          # "text_only" | "image_only" | "text_and_image"

    # ── Set by retrieve node ──
    text_chunks_raw: Optional[str]         # raw string returned by TextRetrievalTool
    image_results_raw: Optional[str]       # raw string returned by image retrieval tool

    # ── Set by generate node ──
    generated_output: Optional[str]        # final differential text from LLM


# ── Node 1: route ─────────────────────────────────────────────────────────────
def route_node(state: OralLesionState) -> OralLesionState:
    """
    Inspect inputs and decide which retrieval path to use.
    Sets state["route_decision"] to one of:
        "text_only"       — description provided, no image
        "image_only"      — image provided, no description
        "text_and_image"  — both provided
    No LLM call here — pure logic on input presence.
    """
    has_text  = bool(state.get("user_description", "").strip())
    has_image = bool(state.get("uploaded_image_path"))

    if has_text and has_image:
        decision = "text_and_image"
    elif has_image:
        decision = "image_only"
    else:
        decision = "text_only"   # default — text_only even if description is thin

    print(f"[route] decision: {decision}")
    return {"route_decision": decision}


# ── Conditional edge function ─────────────────────────────────────────────────
def route_decision(state: OralLesionState) -> str:
    """
    Called by LangGraph to determine which node to go to after route_node.
    Returns the node name as a string — must match keys in add_conditional_edges().
    """
    return state["route_decision"]


# ── Node 2: retrieve ──────────────────────────────────────────────────────────
def retrieve_node(state: OralLesionState) -> OralLesionState:
    """
    Calls the appropriate LangChain retrieval tools based on route_decision.

    text_only:       TextRetrievalTool + ImageTextRetrievalTool
    image_only:      ImageImageRetrievalTool (CLIP image→image)
    text_and_image:  TextRetrievalTool + ImageImageRetrievalTool (best of both)

    All tools return strings — stored as-is in state for the generate node.
    """
    decision    = state["route_decision"]
    description = state.get("user_description", "")
    image_path  = state.get("uploaded_image_path")

    text_chunks_raw  = ""
    image_results_raw = ""

    if decision == "text_only":
        print("[retrieve] text_only — running TextRetrievalTool + ImageTextRetrievalTool")
        text_chunks_raw   = text_retriever_tool.func(description)
        image_results_raw = image_retriever_by_text_tool.func(description)

    elif decision == "image_only":
        print("[retrieve] image_only — running ImageImageRetrievalTool")
        # No text description — use a generic prompt for text retrieval fallback
        image_results_raw = image_retriever_by_image_tool.func(image_path)
        text_chunks_raw   = text_retriever_tool.func("oral lesion differential diagnosis")

    elif decision == "text_and_image":
        print("[retrieve] text_and_image — running all tools")
        text_chunks_raw   = text_retriever_tool.func(description)
        image_results_raw = image_retriever_by_image_tool.func(image_path)

    return {
        "text_chunks_raw":   text_chunks_raw,
        "image_results_raw": image_results_raw,
    }


# ── Node 3: generate ──────────────────────────────────────────────────────────
def generate_node(state: OralLesionState) -> OralLesionState:
    """
    Calls local Ollama vision-LLM with:
        - Retrieved text chunks as grounding context
        - Retrieved image metadata as reference summary
        - User description
        - Uploaded image (base64) if provided

    Returns the generated differential as state["generated_output"].
    """
    description       = state.get("user_description", "lesion shown in image")
    image_path        = state.get("uploaded_image_path")
    text_chunks_raw   = state.get("text_chunks_raw", "")
    image_results_raw = state.get("image_results_raw", "")

    prompt = f"""You are a clinical oral pathology reference assistant helping a junior clinician narrow down a differential diagnosis.

RETRIEVED CLINICAL REFERENCE (use this as your grounding knowledge):
{text_chunks_raw[:3000]}

RETRIEVED SIMILAR REFERENCE IMAGES:
{image_results_raw[:1000]}

CLINICIAN'S DESCRIPTION:
{description}

Based on the retrieved references above, provide:
1. The most likely confusion pair this presentation belongs to
2. Which condition is more likely and why, based on the described features
3. The key clinical test or observation that would confirm or rule out each condition
4. A clear recommendation for next steps

Be concise and clinically focused. Do not diagnose — frame your output as a reference to support the clinician's own judgment.

IMPORTANT: This is an educational reference tool only. Always defer to clinical examination and professional judgment."""

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    # Attach uploaded image as base64 if provided — enables LLaVA vision
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            payload["images"] = [base64.b64encode(f.read()).decode("utf-8")]

    print("[generate] calling Ollama...")
    try:
        response = requests.post(
    OLLAMA_URL,
    json=payload,
    timeout=120,
)

        print("=" * 60)
        print("Status Code:", response.status_code)
        print("Response:")
        print(response.text)
        print("=" * 60)

        response.raise_for_status()

        output = response.json().get("response", "No response from model.")
    except requests.exceptions.ConnectionError:
        output = (
            "Could not connect to Ollama. "
            f"Run `ollama serve` and `ollama pull {OLLAMA_MODEL}` first."
        )
    except Exception as e:
        output = f"Generation error: {e}"

    print("[generate] done")
    return {"generated_output": output}


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Assembles and compiles the LangGraph StateGraph.

    Graph structure:
        START → route_node
                    ↓ (conditional edge on route_decision)
            ┌───────────────────────────┐
            │ text_only                 │
            │ image_only        → retrieve_node → generate_node → END
            │ text_and_image            │
            └───────────────────────────┘
    """
    graph = StateGraph(OralLesionState)

    # Add nodes
    graph.add_node("route",    route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # Entry point
    graph.set_entry_point("route")

    # Conditional edge: route → retrieve (all three paths lead to retrieve)
    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "text_only":       "retrieve",
            "image_only":      "retrieve",
            "text_and_image":  "retrieve",
        }
    )

    # Fixed edges: retrieve → generate → END
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graph = build_graph()
    print("Graph compiled successfully.\n")
    print("Graph nodes:", list(graph.nodes.keys()) if hasattr(graph, 'nodes') else "see above")

    # Test 1 — text only
    print("\n=== Test 1: text only ===")
    result = graph.invoke({
        "user_description": "white patch on buccal mucosa, cannot be wiped off, patient is a heavy smoker",
        "uploaded_image_path": None
    })
    print("\nRoute decision:", result["route_decision"])
    print("\nGenerated output:\n", result["generated_output"])

    # Test 2 — text + image (provide a real image path to test vision)
    print("\n=== Test 2: text + image (skipped — no test image available) ===")
    print("To test: set uploaded_image_path to a real image path")
