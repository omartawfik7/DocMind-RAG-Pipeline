"""
agents/supervisor.py — Supervisor / Planner Agent
=====================================================
Coordinates the Retrieval and Analysis agents. This is deliberately a
bounded, deterministic state machine rather than a free LLM-driven
ReAct loop: the Supervisor itself decides how many retrieval steps to
run (capped at MAX_SUBQUERIES / MAX_SUPERVISOR_STEPS) and never lets an
LLM decide to call an arbitrary tool an arbitrary number of times. That
is itself an excessive-agency mitigation, not just an implementation
simplification -- see docs/INTERVIEW_GUIDE.md for the reasoning.

Every step is checked against the circuit breaker; a triggered kill
switch is caught here and turned into a safe, generic response instead
of propagating as a 500.
"""

import re
import time
import uuid

import config
from agents import retrieval_agent, analysis_agent
from agents.tools import UnauthorizedToolError
from anomaly import RunStats, anomaly_detector
from audit_log import log_event
from circuit_breaker import circuit_breaker, KillSwitchTriggered
from llm_providers import LLMProvider, get_provider
from sandbox import ToolTimeoutError
from security.guardrail import guardrail
from security.output_validator import output_validator


def _plan_subqueries(query: str) -> list:
    """Deterministic query decomposition: split a multi-part question
    into up to MAX_SUBQUERIES pieces on '?' boundaries. Real planning
    logic, intentionally bounded rather than open-ended -- see module
    docstring."""
    parts = [p.strip() for p in re.split(r"(?<=\?)\s+", query) if p.strip()]
    if len(parts) <= 1:
        return [query]
    return parts[: config.MAX_SUBQUERIES]


def _safe_response(reason: str, run_id: str, decision: str = "BLOCK") -> dict:
    return {
        "answer": "I can't help with that request.",
        "sources": [],
        "chunks_retrieved": 0,
        "security": {"decision": decision, "reason": reason},
        "run_id": run_id,
    }


class SupervisorAgent:
    def __init__(self, provider: LLMProvider = None):
        # Provider is injectable for tests (FakeLLMProvider); defaults
        # to whatever LLM_PROVIDER selects in production.
        self._provider = provider

    def _get_provider(self) -> LLMProvider:
        return self._provider or get_provider()

    def run(self, query: str, doc_filter=None, chat_history: list = None, request_id: str = None) -> dict:
        run_id = str(uuid.uuid4())
        stats = RunStats(run_id=run_id)

        log_event(
            run_id=run_id,
            agent_name="supervisor",
            action="RUN_START",
            user_request_id=request_id,
            extra={"query_length": len(query or "")},
        )

        # 1. Input guardrail -- deterministic, runs before any planning.
        input_decision = guardrail.check_input(query)
        log_event(
            run_id=run_id,
            agent_name="security.guardrail",
            action="INPUT_CHECK",
            user_request_id=request_id,
            security_decision=input_decision.action,
            security_reason=input_decision.reason,
        )
        if input_decision.blocked:
            stats.record_block()
            stats.record_injection_attempt()
            anomaly_detector.record_duration(stats.elapsed_seconds())
            return _safe_response(input_decision.reason, run_id, decision="BLOCK")

        try:
            return self._execute(query, doc_filter, chat_history, request_id, run_id, stats, input_decision)
        except KillSwitchTriggered as e:
            anomaly_detector.record_duration(stats.elapsed_seconds())
            return _safe_response(e.reason, run_id, decision="BLOCK")

    def _execute(self, query, doc_filter, chat_history, request_id, run_id, stats, input_decision):
        subqueries = _plan_subqueries(query)

        chunk_map = {}  # (doc_name, chunk_index) -> chunk, dedupes across subqueries
        for subquery in subqueries:
            stats.record_iteration()
            anomaly_result = anomaly_detector.evaluate(stats)
            circuit_breaker.check(stats, anomaly_result, user_request_id=request_id)

            try:
                result = retrieval_agent.retrieve(
                    subquery,
                    run_id=run_id,
                    stats=stats,
                    doc_filter=doc_filter,
                    user_request_id=request_id,
                )
            except (UnauthorizedToolError, ToolTimeoutError, ValueError):
                # Already logged + counted by tools.execute; skip this
                # subquery and let the circuit breaker decide, on the
                # next iteration check, whether the run should continue.
                continue

            for chunk in result.chunks:
                key = (chunk["doc_name"], chunk["chunk_index"])
                if key not in chunk_map or chunk["score"] > chunk_map[key]["score"]:
                    chunk_map[key] = chunk

        chunks = sorted(chunk_map.values(), key=lambda c: c["score"], reverse=True)[:config.MAX_TOOL_CALLS * 6]

        anomaly_result = anomaly_detector.evaluate(stats)
        circuit_breaker.check(stats, anomaly_result, user_request_id=request_id)

        if not chunks:
            log_event(run_id=run_id, agent_name="supervisor", action="RUN_COMPLETE",
                      user_request_id=request_id, success=True,
                      extra={"outcome": "no_documents"})
            anomaly_detector.record_duration(stats.elapsed_seconds())
            return {
                "status": "ok",
                "answer": "No documents have been uploaded yet, or none matched your question. "
                          "Please upload a PDF or text file first.",
                "sources": [],
                "chunks_retrieved": 0,
                "security": {"decision": input_decision.action, "reason": input_decision.reason},
                "run_id": run_id,
            }

        start = time.monotonic()
        try:
            provider = self._get_provider()
            result = analysis_agent.generate_answer(query, chunks, provider, chat_history)
            duration_ms = (time.monotonic() - start) * 1000
            log_event(
                run_id=run_id,
                agent_name="analysis_agent",
                action="GENERATE_ANSWER",
                user_request_id=request_id,
                duration_ms=duration_ms,
                success=True,
            )
        except Exception as e:
            stats.record_failure()
            duration_ms = (time.monotonic() - start) * 1000
            log_event(
                run_id=run_id,
                agent_name="analysis_agent",
                action="GENERATE_ANSWER_FAILED",
                user_request_id=request_id,
                duration_ms=duration_ms,
                success=False,
                error_category=type(e).__name__,
            )
            anomaly_result = anomaly_detector.evaluate(stats)
            circuit_breaker.check(stats, anomaly_result, user_request_id=request_id)
            anomaly_detector.record_duration(stats.elapsed_seconds())
            return _safe_response("The analysis agent failed to generate an answer.", run_id, decision="REVIEW")

        # 2. Output validator -- deterministic, runs after generation.
        output_decision = output_validator.check(result["answer"])
        log_event(
            run_id=run_id,
            agent_name="security.output_validator",
            action="OUTPUT_CHECK",
            user_request_id=request_id,
            security_decision=output_decision.action,
            security_reason=output_decision.reason,
        )
        if output_decision.action == "BLOCK":
            stats.record_block()

        final_answer = output_decision.redacted_text

        log_event(
            run_id=run_id,
            agent_name="supervisor",
            action="RUN_COMPLETE",
            user_request_id=request_id,
            success=True,
            duration_ms=stats.elapsed_seconds() * 1000,
        )
        anomaly_detector.record_duration(stats.elapsed_seconds())

        return {
            "status": "ok",
            "answer": final_answer,
            "sources": result["sources"],
            "chunks_retrieved": len(chunks),
            "security": {"decision": input_decision.action, "reason": input_decision.reason},
            "run_id": run_id,
        }


supervisor = SupervisorAgent()
