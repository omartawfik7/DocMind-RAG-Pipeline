"""
sandbox.py — Execution-isolation helpers for agent tool calls
=================================================================
Honest scope: this environment has no container runtime available
(no Docker), so this module does NOT provide OS-level process
isolation, memory/CPU rlimits, or network firewalling. What it DOES
provide, for real:

  - Wall-clock timeout enforcement: a tool call is run in a worker
    thread, and the orchestrator refuses to wait past the deadline --
    a slow/hung call is treated as a failed call and counted against
    the circuit breaker, even though the underlying thread is not
    forcibly killed (Python cannot safely preempt a running thread).
  - Environment-variable filtering: tools receive a restricted mapping
    built from an explicit allowlist, never raw os.environ, so a tool
    (or a future tool with a bug) cannot read secrets it has no
    business seeing.
  - Filesystem path guarding: any tool that touches the filesystem
    must have its paths checked against an allowlist of roots before
    the operation runs.

See README.md "Sandbox / Execution Design" for the full real-vs-simulated
breakdown -- this docstring intentionally does not oversell what's here.
"""

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Callable

import config

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-sandbox")


class ToolTimeoutError(Exception):
    """Raised when a sandboxed tool call exceeds its wall-clock budget."""


class PathNotAllowedError(Exception):
    """Raised when a tool attempts to touch a path outside its allowlist."""


def run_with_limits(fn: Callable, *args, timeout_s: float = None, **kwargs):
    """
    Run `fn(*args, **kwargs)` in a worker thread and enforce a hard
    wall-clock timeout.

    This is a COOPERATIVE timeout from the caller's point of view: if
    `fn` overruns, this function stops waiting and raises
    ToolTimeoutError, but the underlying thread may continue running in
    the background until it finishes on its own. True preemptive
    termination would require a subprocess or container boundary. The
    orchestrator treats a timeout as a hard failure regardless.
    """
    timeout_s = timeout_s or config.TOOL_TIMEOUT_SECONDS
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeoutError:
        raise ToolTimeoutError(f"Tool call exceeded {timeout_s}s timeout.")


def env_view(allowlist=None) -> dict:
    """Return a restricted view of environment variables -- only keys
    on the allowlist are visible, so tool code never sees the full
    process environment (API keys, secrets, etc.)."""
    allowlist = allowlist if allowlist is not None else config.TOOL_ENV_ALLOWLIST
    return {k: os.environ[k] for k in allowlist if k in os.environ}


def assert_path_allowed(path: str, allowed_roots=None) -> str:
    """
    Resolve `path` to its canonical absolute form and verify it falls
    within one of the allowed root directories. Raises
    PathNotAllowedError otherwise. Returns the resolved path on success.

    Guards against path traversal (`..`, symlinks, absolute paths
    outside the sandbox root).
    """
    allowed_roots = allowed_roots if allowed_roots is not None else config.SANDBOX_ALLOWED_PATHS
    resolved = Path(path).resolve()
    for root in allowed_roots:
        root_resolved = Path(root).resolve()
        try:
            resolved.relative_to(root_resolved)
            return str(resolved)
        except ValueError:
            continue
    raise PathNotAllowedError(f"Path '{path}' is outside the allowed sandbox roots {allowed_roots}.")
