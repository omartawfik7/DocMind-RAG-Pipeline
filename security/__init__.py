"""security — deterministic guardrails for the agentic RAG pipeline.

Everything in this package is plain Python (regex/heuristics), not an
LLM call. That is a deliberate design choice: an LLM asked "is this
malicious?" can itself be fooled by the same injection it is meant to
catch, and its judgment is expensive and non-deterministic. Keeping
the security perimeter deterministic means it is fast, testable, and
its decisions are always reproducible from the same input.
"""
