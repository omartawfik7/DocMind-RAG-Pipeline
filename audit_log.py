"""
audit_log.py — Append-only structured audit logging for agent runs.
=======================================================================
Every security decision and agent action gets one JSONL line in
logs/audit.jsonl (gitignored -- never committed, never contains raw
secrets). This is the observability backbone: anomaly detection reads
recent history from it indirectly via the in-process AnomalyDetector,
and a human debugging a run can `grep run_id logs/audit.jsonl` to see
exactly what happened, in order, with why.
"""

import json
import os
import re
import threading
from datetime import datetime, timezone

import config

_LOCK = threading.Lock()

# Redact anything that looks like a secret before it ever reaches disk,
# even though callers are expected to pass already-sanitized params.
# Defense in depth: a bug upstream should not become a leaked-key
# incident in the audit trail itself.
_SECRET_LIKE = re.compile(
    r"(gsk_[A-Za-z0-9]{15,}|sk-[A-Za-z0-9\-_]{15,}|AKIA[0-9A-Z]{12,}|AIza[0-9A-Za-z\-_]{30,})"
)


def redact(value):
    """Recursively scrub secret-shaped substrings from strings, dicts,
    and lists before they're logged."""
    if isinstance(value, str):
        return _SECRET_LIKE.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def log_event(
    run_id: str,
    agent_name: str,
    action: str,
    *,
    user_request_id: str = None,
    tool: str = None,
    tool_params: dict = None,
    security_decision: str = None,
    security_reason: str = None,
    duration_ms: float = None,
    success: bool = True,
    error_category: str = None,
    extra: dict = None,
):
    """Append one structured audit event. Never raises -- a logging
    failure must not take down a request."""
    event = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_request_id": user_request_id,
        "agent_name": agent_name,
        "action": action,
        "tool": tool,
        "tool_params": redact(tool_params) if tool_params else None,
        "security_decision": security_decision,
        "security_reason": security_reason,
        "duration_ms": duration_ms,
        "success": success,
        "error_category": error_category,
    }
    if extra:
        event.update(redact(extra))

    line = json.dumps(event, default=str)
    try:
        os.makedirs(os.path.dirname(config.AUDIT_LOG_PATH) or ".", exist_ok=True)
        with _LOCK:
            with open(config.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        # Audit logging is best-effort; never let it break the request path.
        pass

    return event
