"""
agents/tools.py — Tool allowlist and sandboxed execution wrapper.
=====================================================================
Excessive-agency control, concretely: this is the ONLY place agent
code can reach a capability. If a tool isn't registered here, the
agent layer cannot call it -- upload/delete/filesystem access are
simply never registered, so there is no runtime check to bypass.

Every call goes through execute(), which enforces: allowlist
membership, per-call parameter validation, a wall-clock timeout via
sandbox.run_with_limits, and audit logging of the outcome. Calling an
unregistered tool name is itself a security event (BLOCK, logged,
counted toward the kill switch) -- not a silent no-op.
"""

import time
from dataclasses import dataclass
from typing import Callable

import config
from anomaly import RunStats
from audit_log import log_event
from sandbox import run_with_limits, ToolTimeoutError
from rag_engine import retrieve as _rag_retrieve


class UnauthorizedToolError(Exception):
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' is not in the allowlist.")


@dataclass
class ToolSpec:
    name: str
    func: Callable
    read_only: bool = True
    requires_approval: bool = False
    timeout_s: float = config.TOOL_TIMEOUT_SECONDS


def _validate_retrieve_params(query: str, top_k: int, doc_filter):
    """Least-privilege parameter validation -- reject malformed or
    out-of-range input before it ever reaches rag_engine."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("retrieve_documents: 'query' must be a non-empty string.")
    if len(query) > 2000:
        raise ValueError("retrieve_documents: 'query' exceeds maximum length (2000 chars).")
    if not isinstance(top_k, int) or not (1 <= top_k <= 20):
        raise ValueError("retrieve_documents: 'top_k' must be an integer between 1 and 20.")
    if doc_filter is not None and not isinstance(doc_filter, str):
        raise ValueError("retrieve_documents: 'doc_filter' must be a string or None.")


def _retrieve_documents(query: str, top_k: int = 6, doc_filter=None):
    _validate_retrieve_params(query, top_k, doc_filter)
    return _rag_retrieve(query, top_k=top_k, doc_filter=doc_filter)


# The complete tool allowlist. This is intentionally minimal: the
# agent layer never gets upload, delete, or filesystem tools -- those
# stay direct, human-authenticated API endpoints (see app.py).
TOOL_REGISTRY = {
    "retrieve_documents": ToolSpec(
        name="retrieve_documents",
        func=_retrieve_documents,
        read_only=True,
        requires_approval=False,
    ),
}


def execute(tool_name: str, stats: RunStats, *, run_id: str, user_request_id: str = None, **kwargs):
    """Execute a registered tool under sandbox limits, updating
    `stats` and writing an audit event regardless of outcome."""
    if tool_name not in TOOL_REGISTRY:
        stats.record_block()
        log_event(
            run_id=run_id,
            agent_name="tools",
            action="TOOL_CALL_BLOCKED",
            user_request_id=user_request_id,
            tool=tool_name,
            tool_params=kwargs,
            security_decision="BLOCK",
            security_reason="Requested tool is not in the allowlist.",
            success=False,
            error_category="unauthorized_tool",
        )
        raise UnauthorizedToolError(tool_name)

    spec = TOOL_REGISTRY[tool_name]
    stats.record_tool_call(tool_name)
    start = time.monotonic()
    try:
        result = run_with_limits(spec.func, timeout_s=spec.timeout_s, **kwargs)
        duration_ms = (time.monotonic() - start) * 1000
        log_event(
            run_id=run_id,
            agent_name="tools",
            action="TOOL_CALL",
            user_request_id=user_request_id,
            tool=tool_name,
            tool_params=kwargs,
            security_decision="ALLOW",
            duration_ms=duration_ms,
            success=True,
        )
        return result
    except ToolTimeoutError as e:
        stats.record_failure()
        duration_ms = (time.monotonic() - start) * 1000
        log_event(
            run_id=run_id,
            agent_name="tools",
            action="TOOL_CALL_FAILED",
            user_request_id=user_request_id,
            tool=tool_name,
            tool_params=kwargs,
            duration_ms=duration_ms,
            success=False,
            error_category="timeout",
        )
        raise
    except Exception as e:
        stats.record_failure()
        duration_ms = (time.monotonic() - start) * 1000
        log_event(
            run_id=run_id,
            agent_name="tools",
            action="TOOL_CALL_FAILED",
            user_request_id=user_request_id,
            tool=tool_name,
            tool_params=kwargs,
            duration_ms=duration_ms,
            success=False,
            error_category=type(e).__name__,
        )
        raise
