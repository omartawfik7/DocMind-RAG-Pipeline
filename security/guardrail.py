"""
security/guardrail.py — Deterministic input/content guardrail.
==================================================================
Classifies user queries (direct injection, before the Supervisor plans
anything) and retrieved document chunks (indirect injection, before the
Analysis Agent sees them) into ALLOW / REVIEW / BLOCK.

This module never calls an LLM. See security/__init__.py for why.
"""

from dataclasses import dataclass, field

from security.patterns import PATTERN_GROUPS, detect_encoded_payload


@dataclass
class Decision:
    action: str            # "ALLOW" | "REVIEW" | "BLOCK"
    category: str          # e.g. "instruction_override", "clean"
    reason: str            # human-readable explanation, safe to log
    matches: list = field(default_factory=list)  # matched pattern categories, for REVIEW aggregation

    @property
    def allowed(self) -> bool:
        return self.action == "ALLOW"

    @property
    def blocked(self) -> bool:
        return self.action == "BLOCK"


def _classify(text: str, source_label: str) -> Decision:
    """Shared classification logic for both user input and retrieved chunks."""
    if not text or not text.strip():
        return Decision("ALLOW", "clean", "Empty input.")

    review_hits = []
    for group in PATTERN_GROUPS:
        for pattern in group["patterns"]:
            if pattern.search(text):
                if group["decision"] == "BLOCK":
                    return Decision(
                        "BLOCK",
                        group["category"],
                        f"{source_label}: {group['description']}",
                        matches=[group["category"]],
                    )
                review_hits.append(group["category"])
                break  # one hit per group is enough

    encoded = detect_encoded_payload(text)
    if encoded:
        category, description = encoded
        review_hits.append(category)

    if review_hits:
        return Decision(
            "REVIEW",
            review_hits[0],
            f"{source_label}: suspicious pattern(s) detected: {', '.join(review_hits)}.",
            matches=review_hits,
        )

    return Decision("ALLOW", "clean", f"{source_label}: no suspicious patterns detected.")


class InputGuardrail:
    """Deterministic classifier for user queries and retrieved content."""

    def check_input(self, query: str) -> Decision:
        """Classify a raw user query before any planning happens."""
        return _classify(query, "user_query")

    def scan_chunk(self, chunk_text: str, doc_name: str) -> Decision:
        """Classify a single retrieved document chunk for indirect
        prompt injection -- text an attacker embedded in a document,
        hoping the model will treat it as an instruction once it's
        pulled into context."""
        return _classify(chunk_text, f"retrieved_chunk[{doc_name}]")


# Module-level singleton -- the guardrail is stateless, so one instance
# is safe to share across requests.
guardrail = InputGuardrail()
