"""
anomaly.py — Rule/statistical anomaly detection for agent runs.
===================================================================
Interpretable-first: every rule below is a plain threshold or a simple
rolling z-score, chosen deliberately over a black-box ML model so that
a triggered anomaly always comes with a human-readable reason. A
lightweight statistical layer (the duration z-score) is the one place
this goes beyond fixed thresholds, and it's the natural second layer
to extend with a real ML detector later -- see README.md "Future
Improvements".

Anomaly scoring feeds the circuit breaker (circuit_breaker.py): once
the aggregated score for a run crosses ANOMALY_SCORE_KILL_THRESHOLD,
the kill switch trips.
"""

import statistics
import threading
import time
from dataclasses import dataclass, field

import config


@dataclass
class RunStats:
    """Mutable per-run counters the Supervisor updates as it works."""
    run_id: str
    start_time: float = field(default_factory=time.monotonic)
    iterations: int = 0
    tool_calls: int = 0
    failures: int = 0
    blocks: int = 0
    injection_attempts: int = 0
    tool_sequence: list = field(default_factory=list)

    def record_iteration(self):
        self.iterations += 1

    def record_tool_call(self, tool_name: str):
        self.tool_calls += 1
        self.tool_sequence.append(tool_name)

    def record_failure(self):
        self.failures += 1

    def record_block(self):
        self.blocks += 1

    def record_injection_attempt(self):
        self.injection_attempts += 1

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time


@dataclass
class AnomalyResult:
    score: float
    reasons: list


class AnomalyDetector:
    """Tracks a rolling window of past run durations (in-process, not
    persisted) to flag abnormally long runs, plus a handful of simple
    rule-based checks against a single run's counters."""

    def __init__(self):
        self._duration_history = []
        self._lock = threading.Lock()

    def record_duration(self, duration_seconds: float):
        with self._lock:
            self._duration_history.append(duration_seconds)
            if len(self._duration_history) > config.ANOMALY_HISTORY_WINDOW:
                self._duration_history.pop(0)

    def _duration_zscore(self, duration_seconds: float):
        with self._lock:
            history = list(self._duration_history)
        if len(history) < 5:
            return None  # not enough history to judge yet
        mean = statistics.mean(history)
        stdev = statistics.pstdev(history)
        if stdev == 0:
            return None
        return (duration_seconds - mean) / stdev

    def _has_repeating_loop(self, tool_sequence: list, min_repeats: int = 3) -> bool:
        """True if the same tool was called `min_repeats` times in a row
        -- a signature of a stuck agentic loop."""
        if len(tool_sequence) < min_repeats:
            return False
        tail = tool_sequence[-min_repeats:]
        return len(set(tail)) == 1

    def evaluate(self, stats: RunStats) -> AnomalyResult:
        reasons = []
        score = 0.0

        if stats.tool_calls > config.ANOMALY_TOOL_CALL_THRESHOLD:
            score += 1
            reasons.append(f"excessive_tool_calls({stats.tool_calls})")

        if stats.failures >= config.ANOMALY_FAILURE_THRESHOLD:
            score += 1
            reasons.append(f"repeated_failures({stats.failures})")

        if stats.blocks >= config.ANOMALY_BLOCK_THRESHOLD:
            score += 1
            reasons.append(f"repeated_blocked_actions({stats.blocks})")

        if stats.injection_attempts >= 2:
            score += 1
            reasons.append(f"repeated_injection_attempts({stats.injection_attempts})")

        if self._has_repeating_loop(stats.tool_sequence):
            score += 1
            reasons.append("unusual_tool_sequence(repeating_loop)")

        duration = stats.elapsed_seconds()
        z = self._duration_zscore(duration)
        if z is not None and z >= config.ANOMALY_DURATION_ZSCORE:
            score += 1
            reasons.append(f"abnormal_duration(z={z:.2f})")

        return AnomalyResult(score=score, reasons=reasons)


anomaly_detector = AnomalyDetector()
