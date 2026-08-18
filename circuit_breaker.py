"""
circuit_breaker.py — Centralized agent kill switch.
======================================================
A single place that decides "this run must stop now" and enforces it.
The Supervisor calls CircuitBreaker.check(...) after every step; if any
condition trips, a KillSwitchTriggered exception aborts the run and a
KILL_SWITCH_TRIGGERED event is written to the audit log with the exact
reason and run_id, before the API returns a safe fallback response.
"""

import config
from anomaly import RunStats, AnomalyResult
from audit_log import log_event


class KillSwitchTriggered(Exception):
    def __init__(self, reason: str, run_id: str):
        self.reason = reason
        self.run_id = run_id
        super().__init__(f"Kill switch triggered for run {run_id}: {reason}")


class CircuitBreaker:
    def check(
        self,
        stats: RunStats,
        anomaly_result: AnomalyResult,
        *,
        prohibited_tool_requested: bool = False,
        user_request_id: str = None,
    ):
        """Raises KillSwitchTriggered (and logs it) if any threshold is
        breached. Returns None if the run may continue."""
        reason = None

        if prohibited_tool_requested:
            reason = "PROHIBITED_TOOL_REQUESTED: agent attempted to invoke a tool outside its allowlist."
        elif stats.iterations > config.KILL_SWITCH_MAX_ITERATIONS:
            reason = f"MAX_ITERATIONS_EXCEEDED: {stats.iterations} > {config.KILL_SWITCH_MAX_ITERATIONS}."
        elif stats.tool_calls > config.KILL_SWITCH_MAX_TOOL_CALLS:
            reason = f"MAX_TOOL_CALLS_EXCEEDED: {stats.tool_calls} > {config.KILL_SWITCH_MAX_TOOL_CALLS}."
        elif stats.blocks >= config.KILL_SWITCH_MAX_SECURITY_VIOLATIONS:
            reason = f"REPEATED_SECURITY_VIOLATIONS: {stats.blocks} blocked actions in this run."
        elif stats.failures >= config.KILL_SWITCH_MAX_FAILURES:
            reason = f"REPEATED_EXECUTION_FAILURES: {stats.failures} failed tool/agent calls."
        elif stats.elapsed_seconds() > config.KILL_SWITCH_MAX_DURATION_SECONDS:
            reason = f"EXECUTION_TIMEOUT_EXCEEDED: run exceeded {config.KILL_SWITCH_MAX_DURATION_SECONDS}s."
        elif anomaly_result.score >= config.ANOMALY_SCORE_KILL_THRESHOLD:
            reason = f"ANOMALY_SCORE_EXCEEDED: score={anomaly_result.score} ({', '.join(anomaly_result.reasons)})."

        if reason:
            log_event(
                run_id=stats.run_id,
                agent_name="circuit_breaker",
                action="KILL_SWITCH_TRIGGERED",
                user_request_id=user_request_id,
                security_decision="BLOCK",
                security_reason=reason,
                success=False,
                error_category="kill_switch",
            )
            raise KillSwitchTriggered(reason=reason, run_id=stats.run_id)


circuit_breaker = CircuitBreaker()
