"""tests/test_circuit_breaker.py — Kill-switch trip conditions."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from anomaly import RunStats, AnomalyResult
from circuit_breaker import circuit_breaker, KillSwitchTriggered


def _clean_anomaly():
    return AnomalyResult(score=0.0, reasons=[])


def test_no_trip_under_thresholds():
    stats = RunStats(run_id="cb-1")
    stats.record_iteration()
    stats.record_tool_call("retrieve_documents")
    # Should not raise.
    circuit_breaker.check(stats, _clean_anomaly())


def test_trip_on_prohibited_tool():
    stats = RunStats(run_id="cb-2")
    with pytest.raises(KillSwitchTriggered) as exc_info:
        circuit_breaker.check(stats, _clean_anomaly(), prohibited_tool_requested=True)
    assert "PROHIBITED_TOOL_REQUESTED" in exc_info.value.reason
    assert exc_info.value.run_id == "cb-2"


def test_trip_on_max_iterations_exceeded():
    stats = RunStats(run_id="cb-3")
    for _ in range(config.KILL_SWITCH_MAX_ITERATIONS + 1):
        stats.record_iteration()
    with pytest.raises(KillSwitchTriggered) as exc_info:
        circuit_breaker.check(stats, _clean_anomaly())
    assert "MAX_ITERATIONS_EXCEEDED" in exc_info.value.reason


def test_trip_on_max_tool_calls_exceeded():
    stats = RunStats(run_id="cb-4")
    for _ in range(config.KILL_SWITCH_MAX_TOOL_CALLS + 1):
        stats.record_tool_call("retrieve_documents")
    with pytest.raises(KillSwitchTriggered) as exc_info:
        circuit_breaker.check(stats, _clean_anomaly())
    assert "MAX_TOOL_CALLS_EXCEEDED" in exc_info.value.reason


def test_trip_on_repeated_security_violations():
    stats = RunStats(run_id="cb-5")
    for _ in range(config.KILL_SWITCH_MAX_SECURITY_VIOLATIONS):
        stats.record_block()
    with pytest.raises(KillSwitchTriggered) as exc_info:
        circuit_breaker.check(stats, _clean_anomaly())
    assert "REPEATED_SECURITY_VIOLATIONS" in exc_info.value.reason


def test_trip_on_repeated_execution_failures():
    stats = RunStats(run_id="cb-6")
    for _ in range(config.KILL_SWITCH_MAX_FAILURES):
        stats.record_failure()
    with pytest.raises(KillSwitchTriggered) as exc_info:
        circuit_breaker.check(stats, _clean_anomaly())
    assert "REPEATED_EXECUTION_FAILURES" in exc_info.value.reason


def test_trip_on_execution_timeout():
    stats = RunStats(run_id="cb-7")
    stats.start_time -= (config.KILL_SWITCH_MAX_DURATION_SECONDS + 1)
    with pytest.raises(KillSwitchTriggered) as exc_info:
        circuit_breaker.check(stats, _clean_anomaly())
    assert "EXECUTION_TIMEOUT_EXCEEDED" in exc_info.value.reason


def test_trip_on_anomaly_score_exceeded():
    stats = RunStats(run_id="cb-8")
    high_anomaly = AnomalyResult(score=config.ANOMALY_SCORE_KILL_THRESHOLD, reasons=["test_reason"])
    with pytest.raises(KillSwitchTriggered) as exc_info:
        circuit_breaker.check(stats, high_anomaly)
    assert "ANOMALY_SCORE_EXCEEDED" in exc_info.value.reason


def test_kill_switch_event_is_logged(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", str(log_path))

    stats = RunStats(run_id="cb-9")
    with pytest.raises(KillSwitchTriggered):
        circuit_breaker.check(stats, _clean_anomaly(), prohibited_tool_requested=True)

    assert log_path.exists()
    content = log_path.read_text()
    assert "KILL_SWITCH_TRIGGERED" in content
    assert "cb-9" in content
