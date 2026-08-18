"""
config.py — Central configuration for the agentic + security layer
=====================================================================
Every threshold that governs agent behavior, security decisions, and
governance lives here so there is exactly one place to look (and tune)
instead of magic numbers scattered across the codebase.

All values are read from environment variables with safe defaults, so
an unmodified .env keeps the app working exactly as before this
upgrade — nothing here is required to run the app.
"""

import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# -- LLM provider selection ----------------------------------------
# "groq" (default, preserves existing deployed behavior), "anthropic", "openai"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# -- Excessive-agency controls (Supervisor) -------------------------
MAX_SUPERVISOR_STEPS = _int_env("MAX_SUPERVISOR_STEPS", 6)
MAX_TOOL_CALLS = _int_env("MAX_TOOL_CALLS", 4)
MAX_SUBQUERIES = _int_env("MAX_SUBQUERIES", 3)
TOOL_TIMEOUT_SECONDS = _float_env("TOOL_TIMEOUT_SECONDS", 15.0)

# -- Anomaly detection thresholds -----------------------------------
ANOMALY_TOOL_CALL_THRESHOLD = _int_env("ANOMALY_TOOL_CALL_THRESHOLD", MAX_TOOL_CALLS)
ANOMALY_FAILURE_THRESHOLD = _int_env("ANOMALY_FAILURE_THRESHOLD", 3)
ANOMALY_BLOCK_THRESHOLD = _int_env("ANOMALY_BLOCK_THRESHOLD", 2)
ANOMALY_DURATION_ZSCORE = _float_env("ANOMALY_DURATION_ZSCORE", 3.0)
ANOMALY_SCORE_KILL_THRESHOLD = _float_env("ANOMALY_SCORE_KILL_THRESHOLD", 3.0)
ANOMALY_HISTORY_WINDOW = _int_env("ANOMALY_HISTORY_WINDOW", 50)

# -- Kill switch / circuit breaker thresholds -----------------------
KILL_SWITCH_MAX_ITERATIONS = _int_env("KILL_SWITCH_MAX_ITERATIONS", MAX_SUPERVISOR_STEPS)
KILL_SWITCH_MAX_TOOL_CALLS = _int_env("KILL_SWITCH_MAX_TOOL_CALLS", MAX_TOOL_CALLS)
KILL_SWITCH_MAX_SECURITY_VIOLATIONS = _int_env("KILL_SWITCH_MAX_SECURITY_VIOLATIONS", 2)
KILL_SWITCH_MAX_FAILURES = _int_env("KILL_SWITCH_MAX_FAILURES", 3)
KILL_SWITCH_MAX_DURATION_SECONDS = _float_env("KILL_SWITCH_MAX_DURATION_SECONDS", 60.0)

# -- Filesystem / environment restrictions for sandboxed tool calls --
# Tools never receive the raw process environment -- only these keys,
# if present, are made visible to them via sandbox.env_view().
TOOL_ENV_ALLOWLIST = ["EMBED_MODEL", "GROQ_MODEL"]

# Paths tool code is allowed to touch (currently no tool touches the
# filesystem, but this is the enforcement point for any that do).
SANDBOX_ALLOWED_PATHS = [os.path.abspath("./uploads")]

# -- Audit log ---------------------------------------------------------
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "./logs/audit.jsonl")
