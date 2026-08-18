"""tests/test_anomaly.py — Rule-based anomaly detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from anomaly import AnomalyDetector, RunStats


def test_clean_run_has_zero_score():
    detector = AnomalyDetector()
    stats = RunStats(run_id="an-1")
    stats.record_iteration()
    stats.record_tool_call("retrieve_documents")
    result = detector.evaluate(stats)
    assert result.score == 0
    assert result.reasons == []


def test_excessive_tool_calls_flagged():
    detector = AnomalyDetector()
    stats = RunStats(run_id="an-2")
    for _ in range(config.ANOMALY_TOOL_CALL_THRESHOLD + 1):
        stats.record_tool_call("retrieve_documents")
    result = detector.evaluate(stats)
    assert result.score >= 1
    assert any("excessive_tool_calls" in r for r in result.reasons)


def test_repeated_failures_flagged():
    detector = AnomalyDetector()
    stats = RunStats(run_id="an-3")
    for _ in range(config.ANOMALY_FAILURE_THRESHOLD):
        stats.record_failure()
    result = detector.evaluate(stats)
    assert any("repeated_failures" in r for r in result.reasons)


def test_repeated_blocks_flagged():
    detector = AnomalyDetector()
    stats = RunStats(run_id="an-4")
    for _ in range(config.ANOMALY_BLOCK_THRESHOLD):
        stats.record_block()
    result = detector.evaluate(stats)
    assert any("repeated_blocked_actions" in r for r in result.reasons)


def test_repeated_injection_attempts_flagged():
    detector = AnomalyDetector()
    stats = RunStats(run_id="an-5")
    stats.record_injection_attempt()
    stats.record_injection_attempt()
    result = detector.evaluate(stats)
    assert any("repeated_injection_attempts" in r for r in result.reasons)


def test_repeating_tool_sequence_flagged_as_loop():
    detector = AnomalyDetector()
    stats = RunStats(run_id="an-6")
    for _ in range(3):
        stats.record_tool_call("retrieve_documents")
    result = detector.evaluate(stats)
    assert any("unusual_tool_sequence" in r for r in result.reasons)


def test_abnormal_duration_flagged_after_history_builds():
    detector = AnomalyDetector()
    # Seed a history of short, consistent run durations.
    for d in [1.0, 1.1, 0.9, 1.0, 1.05]:
        detector.record_duration(d)

    stats = RunStats(run_id="an-7")
    stats.start_time -= 30  # simulate a run that has been running for 30s
    result = detector.evaluate(stats)
    assert any("abnormal_duration" in r for r in result.reasons)


def test_no_duration_flag_without_enough_history():
    detector = AnomalyDetector()
    stats = RunStats(run_id="an-8")
    stats.start_time -= 30
    result = detector.evaluate(stats)
    assert not any("abnormal_duration" in r for r in result.reasons)
