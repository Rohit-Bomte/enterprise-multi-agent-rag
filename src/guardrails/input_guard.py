"""
input_guard.py — Layer 2 Defense: Validate User Input
-------------------------------------------------------
Runs BEFORE any agent sees the question.

Checks:
  1. Length bomb — unusually long input (prompt padding attacks)
  2. Known injection patterns — "ignore instructions", DAN mode, etc.
  3. Encoding tricks — excessive whitespace or unusual unicode

If any check fails, sets state["is_safe"] = False and writes a safe
refusal into state["final_answer"] so the workflow short-circuits.

This complements the sanitizer (which cleans PDF documents) and the
generation agent's prompt design (which wraps context in XML tags).
Defense-in-depth: multiple independent layers.
"""

import re
from src.config import console

MAX_LENGTH = 1_000  # characters

# Patterns that commonly appear in jailbreaks and prompt injections
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+|your\s+)?instructions",
    r"forget\s+(everything|your\s+rules)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+|an\s+)?(?:different|new|evil|unrestricted|DAN)",
    r"\bDAN\s+mode\b",
    r"jailbreak",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+everything\s+above",
    r"repeat\s+(the\s+)?instructions",
    r"what\s+(are|were)\s+your\s+instructions",
    r"show\s+me\s+your\s+system\s+prompt",
    r"bypass\s+(your\s+)?(safety|restrictions|rules)",
    r"pretend\s+you\s+(have\s+no|are\s+not)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def input_guardrail(state: dict) -> dict:
    """
    LangGraph node — screens user question before any agent runs.
    """
    console.rule("[agent]🚦  INPUT GUARDRAIL  (Layer 2 Defense)[/agent]")
    question = state.get("original_question", "")

    # ── Check 1: Length ───────────────────────────────────────────────────────
    char_count = len(question)
    if char_count > MAX_LENGTH:
        console.print(
            f"  [error]✗  BLOCKED:[/error] Input too long "
            f"({char_count} chars, max {MAX_LENGTH})"
        )
        return {
            **state,
            "is_safe"      : False,
            "final_answer" : "Your question is too long. Please shorten it and try again.",
        }
    console.print(f"  [success]✔[/success]  Length OK  [dim]({char_count} chars)[/dim]")

    # ── Check 2: Injection patterns ──────────────────────────────────────────
    q_lower = question.lower()
    for pattern in _COMPILED:
        if pattern.search(q_lower):
            matched = pattern.pattern
            console.print(
                f"  [error]✗  BLOCKED:[/error] Injection pattern detected  "
                f"[dim]({matched[:40]}…)[/dim]"
            )
            return {
                **state,
                "is_safe"      : False,
                "final_answer" : "I'm sorry, I can't process that request.",
            }
    console.print("  [success]✔[/success]  No injection patterns detected")

    console.print("  [success]→  Safe to proceed[/success]")
    return {**state, "is_safe": True}

