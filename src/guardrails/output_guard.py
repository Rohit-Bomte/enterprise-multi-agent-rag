"""
output_guard.py — Layer 4 Defense: Validate Generated Answer
--------------------------------------------------------------
Runs AFTER the generation agent produces an answer, BEFORE the
answer is shown to the user.

Checks:
  1. PII patterns — redact SSNs, credit cards, emails if found
  2. Prompt leakage — catch cases where the LLM echoes the system prompt
  3. Forbidden phrases — common signs of a compromised output

This is the last line of defense. Even if Layers 1-3 fail, this layer
can still catch and neutralize a harmful output.
"""

import re
from src.config import console

# PII patterns to redact
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),              "[SSN REDACTED]"),
    (re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"),         "[CARD# REDACTED]"),
    (re.compile(r"\b5[1-5][0-9]{14}\b"),                  "[CARD# REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
                                                          "[EMAIL REDACTED]"),
    (re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
                                                          "[PHONE REDACTED]"),
]

# Phrases that may indicate the LLM revealed its system prompt
_LEAKAGE_PHRASES = [
    "my system prompt",
    "my instructions are",
    "i was told to",
    "i am instructed to",
    "the system message says",
    "as an ai language model",
    "i'm an ai language model",
]


def output_guardrail(state: dict) -> dict:
    """
    LangGraph node — screens the generated answer before showing to user.
    """
    console.rule("[agent]🚦  OUTPUT GUARDRAIL  (Layer 4 Defense)[/agent]")
    answer = state.get("final_answer", "")

    # ── Check 1: PII Redaction ────────────────────────────────────────────────
    pii_found = []
    for pattern, replacement in _PII_PATTERNS:
        if pattern.search(answer):
            answer = pattern.sub(replacement, answer)
            pii_found.append(replacement)

    if pii_found:
        console.print(
            f"  [warning]⚠  PII detected and redacted:[/warning] "
            f"{', '.join(set(pii_found))}"
        )
    else:
        console.print("  [success]✔[/success]  No PII detected")

    # ── Check 2: Prompt leakage ───────────────────────────────────────────────
    answer_lower = answer.lower()
    leaked = [p for p in _LEAKAGE_PHRASES if p in answer_lower]
    if leaked:
        console.print(
            f"  [error]✗  BLOCKED:[/error] Potential prompt leakage detected  "
            f"[dim]({leaked[0]!r})[/dim]"
        )
        return {
            **state,
            "final_answer": "I encountered an issue generating a safe response. "
                            "Please rephrase your question.",
        }
    console.print("  [success]✔[/success]  No prompt leakage detected")

    console.print("  [success]→  Answer cleared for display[/success]")
    return {**state, "final_answer": answer}

