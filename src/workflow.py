"""
workflow.py — LangGraph Multi-Agent StateGraph
------------------------------------------------
Wires all five nodes into a sequential pipeline:

  START
    │
    ▼
  [input_guardrail]  — screens user question (Layer 2 defense)
    │
    ▼
  [research_agent]   — rewrites question into effective search query
    │
    ▼
  [retrieval_agent]  — hybrid search (dense+BM25+RRF) + reranking
    │
    ▼
  [generation_agent] — synthesizes cited answer from retrieved chunks
    │
    ▼
  [output_guardrail] — screens generated answer (Layer 4 defense)
    │
    ▼
  END

Shared State (TypedDict):
  All nodes read from and write to this single dictionary.
  LangGraph passes it automatically between nodes.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from src.guardrails.input_guard      import input_guardrail
from src.guardrails.output_guard     import output_guardrail
from src.agents.research_agent       import run_research_agent
from src.agents.retrieval_agent      import run_retrieval_agent
from src.agents.generation_agent     import run_generation_agent


# ── Shared State Schema ────────────────────────────────────────────────────────
class RAGState(TypedDict):
    # Set by caller
    original_question : str

    # Set by input_guardrail
    is_safe           : bool

    # Set by research_agent
    rewritten_query   : str

    # Set by retrieval_agent
    reranked_chunks   : list

    # Set by generation_agent → modified by output_guardrail
    final_answer      : str


# ── Build the Graph ────────────────────────────────────────────────────────────
def build_workflow():
    """
    Construct and compile the LangGraph StateGraph.

    Returns:
        A compiled LangGraph app ready for .invoke(state) calls.
    """
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("input_guard",    input_guardrail)
    graph.add_node("research",       run_research_agent)
    graph.add_node("retrieval",      run_retrieval_agent)
    graph.add_node("generation",     run_generation_agent)
    graph.add_node("output_guard",   output_guardrail)

    # Conditional routing logic
    def route_after_input_guard(state: RAGState) -> str:
        """If input guardrail detects a threat, skip directly to output guardrail."""
        if not state.get("is_safe", True):
            return "output_guard"
        return "research"

    def route_after_research(state: RAGState) -> str:
        """If research agent detects malicious intent, skip retrieval & generation."""
        if state.get("rewritten_query") == "REFUSE":
            return "output_guard"
        return "retrieval"

    # Edges
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {"output_guard": "output_guard", "research": "research"}
    )
    graph.add_conditional_edges(
        "research",
        route_after_research,
        {"output_guard": "output_guard", "retrieval": "retrieval"}
    )
    graph.add_edge("retrieval",    "generation")
    graph.add_edge("generation",   "output_guard")
    graph.add_edge("output_guard", END)

    return graph.compile()


# Singleton — imported by main.py and tests
app = build_workflow()


def ask(question: str) -> str:
    """
    Public API: ask a question and get a cited answer.

    Args:
        question: Natural-language question from the user

    Returns:
        str — The final (guardrail-approved) answer
    """
    initial_state: RAGState = {
        "original_question" : question,
        "is_safe"           : True,
        "rewritten_query"   : "",
        "reranked_chunks"   : [],
        "final_answer"      : "",
    }
    result = app.invoke(initial_state)
    return result["final_answer"]

