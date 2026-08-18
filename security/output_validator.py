"""
security/output_validator.py — Deterministic output guardrail.
==================================================================
Scans a generated answer before it is returned to the user, looking
for two things an LLM could accidentally (or be tricked into) leaking:

  1. Secret-shaped strings (API keys, tokens) -- these should never
     appear in a RAG answer about a user's documents, so any hit is a
     leak, not a false positive worth tolerating.
  2. System-prompt echoes -- the model reciting its own instructions
     back, usually the visible result of a successful prompt-injection
     attempt upstream that the input/chunk guardrails didn't catch.

Like the input guardrail, this is plain pattern matching -- no LLM call.
"""

import re
from dataclasses import dataclass

from security.patterns import SECRET_SHAPE_PATTERNS

_SYSTEM_PROMPT_ECHO = re.compile(
    r"(you are a precise document analyst|answer only from the provided context|"
    r"my system prompt (is|says)|my instructions (are|say))",
    re.IGNORECASE,
)


@dataclass
class OutputDecision:
    action: str        # "ALLOW" | "BLOCK"
    reason: str
    redacted_text: str  # answer with any secret-shaped substrings redacted


class OutputValidator:
    def check(self, answer: str) -> OutputDecision:
        if not answer:
            return OutputDecision("ALLOW", "Empty answer.", answer)

        redacted = answer
        leaked = False
        for pattern in SECRET_SHAPE_PATTERNS:
            if pattern.search(redacted):
                leaked = True
                redacted = pattern.sub("[REDACTED]", redacted)

        if leaked:
            return OutputDecision(
                "BLOCK",
                "Answer contained a secret-shaped string (API key/token pattern) -- redacted before returning.",
                redacted,
            )

        if _SYSTEM_PROMPT_ECHO.search(answer):
            return OutputDecision(
                "BLOCK",
                "Answer appears to echo the system prompt -- likely a successful injection upstream.",
                "I can't share that. Please ask a question about the uploaded documents.",
            )

        return OutputDecision("ALLOW", "No leaked secrets or system-prompt echoes detected.", answer)


output_validator = OutputValidator()
