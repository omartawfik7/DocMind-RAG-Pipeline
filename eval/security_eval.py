"""
eval/security_eval.py — Adversarial security evaluation harness
====================================================================
Runs the same adversarial suite that backs tests/test_security_guardrail.py
through the FULL agentic pipeline (Supervisor -> guardrail -> planning ->
Retrieval Agent -> Analysis Agent -> output validator), using whatever
LLM_PROVIDER/.env is actually configured. This is an integration-level
report, not just a unit-test pass/fail count -- BLOCK-tier attacks never
leave the input guardrail (no network calls), while REVIEW-tier attacks
and legitimate requests exercise the real retrieval + LLM call.

Produces eval/security_eval_report.json with the metrics required by
the security spec: attacks_attempted, attacks_blocked, false_positives,
tool_policy_violations_prevented, kill_switches_triggered,
legitimate_requests_completed.

Run from the repo root:
    python eval/security_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config
from agents.supervisor import supervisor
from agents.tools import execute as execute_tool, UnauthorizedToolError
from anomaly import RunStats, AnomalyResult
from circuit_breaker import circuit_breaker, KillSwitchTriggered
from tests.test_security_guardrail import (  # noqa: E402 -- reuses the pytest-owned case data
    NORMAL_QUERIES, DIRECT_INJECTION, SYSTEM_PROMPT_EXTRACTION, SECRET_EXTRACTION,
    EXCESSIVE_AGENCY, TOOL_MANIPULATION, ENCODED_PAYLOAD, MALFORMED,
)

BLOCK_TIER_ATTACKS = DIRECT_INJECTION + SYSTEM_PROMPT_EXTRACTION + SECRET_EXTRACTION + TOOL_MANIPULATION
REVIEW_TIER_ATTACKS = EXCESSIVE_AGENCY + ENCODED_PAYLOAD
LEGITIMATE = NORMAL_QUERIES + MALFORMED  # malformed input from a real user, not an attacker


def run_input_level_cases():
    results = {
        "attacks_attempted": 0,
        "attacks_blocked": 0,
        "false_positives": 0,
        "legitimate_requests_completed": 0,
        "cases": [],
    }

    for category, text in BLOCK_TIER_ATTACKS + REVIEW_TIER_ATTACKS:
        results["attacks_attempted"] += 1
        try:
            outcome = supervisor.run(text)
            blocked = outcome["security"]["decision"] == "BLOCK"
        except Exception as e:
            blocked = False
            outcome = {"error": str(e)}
        if blocked:
            results["attacks_blocked"] += 1
        results["cases"].append({"category": category, "type": "attack", "text": text, "blocked": blocked})

    for category, text in LEGITIMATE:
        try:
            outcome = supervisor.run(text)
            decision = outcome["security"]["decision"]
            completed = outcome.get("status", "ok") == "ok"
        except Exception:
            decision, completed = "ERROR", False
        if decision == "BLOCK":
            results["false_positives"] += 1
        if completed:
            results["legitimate_requests_completed"] += 1
        results["cases"].append({"category": category, "type": "legitimate", "text": text, "decision": decision})

    return results


def run_tool_policy_cases():
    """Directly attempt to invoke tools outside the allowlist -- these
    must never reach a real implementation."""
    prevented = 0
    unauthorized_tool_names = [
        "delete_all_documents", "read_arbitrary_file", "execute_shell_command", "upload_document",
    ]
    for name in unauthorized_tool_names:
        stats = RunStats(run_id=f"eval-tool-{name}")
        try:
            execute_tool(name, stats, run_id=stats.run_id, query="x")
        except UnauthorizedToolError:
            prevented += 1
    return prevented


def run_kill_switch_cases():
    """Construct one scenario per kill-switch trigger condition and
    confirm the circuit breaker actually trips for each."""
    triggered = 0
    scenarios = []

    s1 = RunStats(run_id="eval-ks-prohibited-tool")
    scenarios.append((s1, {"prohibited_tool_requested": True}))

    s2 = RunStats(run_id="eval-ks-max-iterations")
    for _ in range(config.KILL_SWITCH_MAX_ITERATIONS + 1):
        s2.record_iteration()
    scenarios.append((s2, {}))

    s3 = RunStats(run_id="eval-ks-max-tool-calls")
    for _ in range(config.KILL_SWITCH_MAX_TOOL_CALLS + 1):
        s3.record_tool_call("retrieve_documents")
    scenarios.append((s3, {}))

    s4 = RunStats(run_id="eval-ks-security-violations")
    for _ in range(config.KILL_SWITCH_MAX_SECURITY_VIOLATIONS):
        s4.record_block()
    scenarios.append((s4, {}))

    s5 = RunStats(run_id="eval-ks-execution-failures")
    for _ in range(config.KILL_SWITCH_MAX_FAILURES):
        s5.record_failure()
    scenarios.append((s5, {}))

    s6 = RunStats(run_id="eval-ks-timeout")
    s6.start_time -= (config.KILL_SWITCH_MAX_DURATION_SECONDS + 1)
    scenarios.append((s6, {}))

    for stats, kwargs in scenarios:
        try:
            circuit_breaker.check(stats, AnomalyResult(score=0, reasons=[]), **kwargs)
        except KillSwitchTriggered:
            triggered += 1

    return triggered


def run_eval():
    print("Running adversarial security evaluation against the live pipeline...")
    input_results = run_input_level_cases()
    tool_policy_prevented = run_tool_policy_cases()
    kill_switches = run_kill_switch_cases()

    report = {
        "attacks_attempted": input_results["attacks_attempted"],
        "attacks_blocked": input_results["attacks_blocked"],
        "false_positives": input_results["false_positives"],
        "tool_policy_violations_prevented": tool_policy_prevented,
        "kill_switches_triggered": kill_switches,
        "legitimate_requests_completed": input_results["legitimate_requests_completed"],
        "legitimate_requests_total": len(LEGITIMATE),
        "notes": (
            "attacks_blocked counts only BLOCK-tier decisions (direct injection, secret/"
            "system-prompt extraction, tool manipulation). Excessive-agency and encoded-"
            "payload cases are REVIEW-tier by design -- flagged and logged, not blocked. "
            "See README.md Limitations for why there is no human-review workflow yet."
        ),
        "cases": input_results["cases"],
    }

    out_path = Path(__file__).resolve().parent / "security_eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print(f"\nSaved report to {out_path}")
    return report


if __name__ == "__main__":
    run_eval()
