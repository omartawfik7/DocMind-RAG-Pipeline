"""
security/patterns.py — Regex/heuristic library for prompt-injection and
secret-extraction detection.
==========================================================================
This is the signature database the guardrail and output validator match
against. Every pattern group has:
  - category:    short machine-readable name (used in audit logs)
  - decision:    "BLOCK" or "REVIEW" -- how the guardrail should treat a hit
  - description: human-readable reason, logged alongside the decision
  - patterns:    compiled regexes, matched case-insensitively

Patterns are intentionally broad rather than exhaustive -- this is a
defense-in-depth layer, not a claim of catching every possible phrasing.
See docs/INTERVIEW_GUIDE.md and README.md for the threat model and the
explicit statement that this is not a substitute for least-privilege
tool design (which is the primary defense).
"""

import re


def _c(*patterns: str) -> list:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


PATTERN_GROUPS = [
    {
        "category": "instruction_override",
        "decision": "BLOCK",
        "description": "Attempt to override or discard prior system/developer instructions.",
        "patterns": _c(
            r"ignore (all|any|the)?\s*(previous|prior|above|earlier)\s*(instructions|rules|prompts?|context)",
            r"disregard (all|any|the)?\s*(previous|prior|above|earlier)\s*(instructions|rules|prompts?|context)",
            r"forget (all|everything|your instructions|what (i|you) (said|told))",
            r"new\s+instructions?\s*:",
            r"override\s+(your|the)\s+(instructions|rules|system prompt|programming)",
            r"you are now\s+(in\s+)?(developer|debug|admin|jailbreak|dan|unrestricted)\s*mode",
            r"\bdan\b.{0,20}(mode|prompt|jailbreak)",
            r"act as (if you (have|had) no|an unrestricted|a different ai)",
            r"pretend (you have no|there are no) (restrictions|rules|guidelines|filters)",
            r"from now on,?\s*(you|ignore|disregard)",
            r"do not (follow|obey|apply)\s+(your|the)\s+(previous|original)\s+(instructions|rules)",
        ),
    },
    {
        "category": "system_prompt_extraction",
        "decision": "BLOCK",
        "description": "Attempt to reveal the system prompt or internal configuration.",
        "patterns": _c(
            r"(show|reveal|print|repeat|output|display|leak)\s+(me\s+)?(your|the)\s+(system\s+prompt|initial\s+prompt|instructions)",
            r"what (is|are)\s+your\s+(system\s+prompt|instructions|rules)",
            r"repeat (the|your)\s+(text|words|instructions)\s+above",
            r"(dump|export)\s+(your|the)\s+(configuration|config|prompt|instructions)",
            r"tell me (exactly )?what your (system prompt|instructions) (say|is|are)",
            r"print everything (above|before) this (line|message)",
        ),
    },
    {
        "category": "secret_extraction",
        "decision": "BLOCK",
        "description": "Attempt to extract API keys, credentials, or environment variables.",
        "patterns": _c(
            r"(show|reveal|print|give|leak|dump)\s+(me\s+)?(the|your)\s+(api\s*key|secret\s*key|password|credentials?|token)",
            r"what\s+is\s+(the|your)\s+(groq|qdrant|anthropic|openai)?\s*api[_\s]?key",
            r"(print|dump|show|list)\s+(the\s+)?environment\s+variables?",
            r"(cat|print|show)\b.{0,40}\.env\b",
            r"os\.environ",
            r"process\.env",
            r"getenv\s*\(",
        ),
    },
    {
        "category": "tool_manipulation",
        "decision": "BLOCK",
        "description": "Attempt to invoke, manipulate, or escape allowed tool boundaries.",
        "patterns": _c(
            r"\b(run|execute)\s+(this\s+)?(shell|bash|command|script)",
            r"\bsubprocess\b|\bos\.system\(|\beval\(|\bexec\(",
            r"\brm\s+-rf\b",
            r"curl\s+https?://|wget\s+https?://",
            r"delete\s+all\s+documents?|drop\s+(the\s+)?(collection|table|database)",
            r"call\s+(the\s+)?(delete|upload|admin)\s+tool",
            r"grant\s+(yourself|me)\s+(admin|root|full)\s+access",
        ),
    },
    {
        "category": "excessive_agency",
        "decision": "REVIEW",
        "description": "Language pushing the agent toward unbounded/autonomous execution.",
        "patterns": _c(
            r"(keep|loop|repeat)\s+(trying|going|running)\s+(forever|until it works|indefinitely)",
            r"do\s+whatever\s+it\s+takes",
            r"ignore\s+(your|any)\s+(limits|constraints|step\s+limit|rate\s+limit)",
            r"bypass\s+(your|the)\s+(restrictions|safeguards|limits|guardrails)",
            r"take\s+any\s+action\s+necessary",
            r"without\s+(asking|confirmation|approval)",
        ),
    },
]


# -- Encoded-payload detection ---------------------------------------------
# A long base64-looking or hex-escaped run inside otherwise plain text is
# suspicious -- it's a common way to smuggle instructions past naive
# keyword filters. This is a heuristic (REVIEW), not proof of an attack.
_BASE64_RUN = re.compile(r"(?:[A-Za-z0-9+/]{4}){12,}={0,2}")
_HEX_ESCAPE_RUN = re.compile(r"(?:\\x[0-9a-fA-F]{2}){8,}")
_ZERO_WIDTH_CHARS = re.compile(r"[​‌‍⁠﻿]")


def detect_encoded_payload(text: str):
    """Returns a (category, description) tuple if an encoded-payload
    heuristic fires, else None."""
    if _ZERO_WIDTH_CHARS.search(text):
        return ("encoded_payload", "Zero-width/invisible Unicode characters detected (possible hidden instruction).")
    if _HEX_ESCAPE_RUN.search(text):
        return ("encoded_payload", "Long hex-escape sequence detected (possible encoded instruction payload).")
    m = _BASE64_RUN.search(text)
    if m and len(m.group(0)) >= 48:
        return ("encoded_payload", "Long base64-looking string detected (possible encoded instruction payload).")
    return None


# -- Secret-shaped strings (for output-leak scanning) ----------------------
SECRET_SHAPE_PATTERNS = _c(
    r"gsk_[A-Za-z0-9]{20,}",          # Groq
    r"sk-(ant-)?[A-Za-z0-9\-_]{20,}", # Anthropic / OpenAI-style
    r"AKIA[0-9A-Z]{16}",              # AWS access key id
    r"AIza[0-9A-Za-z\-_]{35}",        # Google API key
    r"qdrant[_-]?api[_-]?key\s*[:=]\s*\S+",
)
