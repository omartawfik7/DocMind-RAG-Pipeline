"""
tests/test_tools_and_sandbox.py — Excessive-agency and isolation tests.

Covers: unauthorized tool requests, malicious/malformed tool
parameters, timeout enforcement, environment-variable filtering, and
filesystem path guarding.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agents.tools import execute, UnauthorizedToolError, TOOL_REGISTRY
from anomaly import RunStats
from sandbox import run_with_limits, ToolTimeoutError, env_view, assert_path_allowed, PathNotAllowedError


def test_only_retrieve_documents_is_registered():
    # Least-privilege: upload/delete/filesystem tools must never be
    # reachable from the agent layer.
    assert set(TOOL_REGISTRY.keys()) == {"retrieve_documents"}
    assert TOOL_REGISTRY["retrieve_documents"].read_only is True


def test_unauthorized_tool_request_is_blocked_and_counted():
    stats = RunStats(run_id="test-run-1")
    with pytest.raises(UnauthorizedToolError):
        execute("delete_all_documents", stats, run_id="test-run-1", doc_name="x")
    assert stats.blocks == 1


def test_unregistered_tool_request_does_not_execute_anything():
    stats = RunStats(run_id="test-run-2")
    with pytest.raises(UnauthorizedToolError):
        execute("read_arbitrary_file", stats, run_id="test-run-2", path="/etc/passwd")
    # Confirms the call never reached spec.func -- no tool_calls recorded,
    # only a block.
    assert stats.tool_calls == 0
    assert stats.blocks == 1


@pytest.mark.parametrize("bad_top_k", [0, -1, 21, "six", None])
def test_malicious_tool_parameters_are_rejected(bad_top_k):
    stats = RunStats(run_id="test-run-3")
    with pytest.raises(ValueError):
        execute(
            "retrieve_documents",
            stats,
            run_id="test-run-3",
            query="a legitimate question",
            top_k=bad_top_k,
        )


def test_empty_query_parameter_is_rejected():
    stats = RunStats(run_id="test-run-4")
    with pytest.raises(ValueError):
        execute("retrieve_documents", stats, run_id="test-run-4", query="   ", top_k=6)


def test_oversized_query_parameter_is_rejected():
    stats = RunStats(run_id="test-run-5")
    with pytest.raises(ValueError):
        execute("retrieve_documents", stats, run_id="test-run-5", query="a" * 3000, top_k=6)


def test_sandbox_timeout_is_enforced():
    def slow_fn():
        time.sleep(2)
        return "done"

    with pytest.raises(ToolTimeoutError):
        run_with_limits(slow_fn, timeout_s=0.1)


def test_sandbox_fast_call_succeeds_within_timeout():
    def fast_fn(x):
        return x * 2

    assert run_with_limits(fast_fn, 21, timeout_s=1.0) == 42


def test_env_view_only_exposes_allowlisted_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_should_not_be_visible")
    monkeypatch.setenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    view = env_view(allowlist=["EMBED_MODEL"])
    assert "EMBED_MODEL" in view
    assert "GROQ_API_KEY" not in view


def test_path_guard_allows_paths_inside_root(tmp_path):
    allowed = [str(tmp_path)]
    inner = tmp_path / "uploads" / "doc.pdf"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.touch()
    resolved = assert_path_allowed(str(inner), allowed_roots=allowed)
    assert resolved.startswith(str(tmp_path.resolve()))


def test_path_guard_blocks_traversal_outside_root(tmp_path):
    allowed = [str(tmp_path / "uploads")]
    with pytest.raises(PathNotAllowedError):
        assert_path_allowed(str(tmp_path / "uploads" / ".." / ".." / "etc" / "passwd"), allowed_roots=allowed)


def test_path_guard_blocks_absolute_path_outside_root(tmp_path):
    allowed = [str(tmp_path / "uploads")]
    with pytest.raises(PathNotAllowedError):
        assert_path_allowed("/etc/passwd", allowed_roots=allowed)
