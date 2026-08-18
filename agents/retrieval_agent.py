"""
agents/retrieval_agent.py — Retrieval Agent
==============================================
Searches the vector database via the sandboxed `retrieve_documents`
tool and returns chunks with their source/metadata. This is also
where indirect prompt injection is defended against: every retrieved
chunk is scanned by the same deterministic guardrail used on user
input, BEFORE it is handed to the Analysis Agent. A chunk that trips
a BLOCK-level pattern (e.g. "ignore previous instructions" embedded
in a document) is dropped from the context entirely -- the model
never sees it.
"""

from dataclasses import dataclass

from agents.tools import execute as execute_tool
from anomaly import RunStats
from audit_log import log_event
from security.guardrail import guardrail


@dataclass
class RetrievalResult:
    chunks: list          # clean chunks, safe to pass to the Analysis Agent
    flagged_count: int    # how many chunks were dropped for indirect injection


def retrieve(
    query: str,
    *,
    run_id: str,
    stats: RunStats,
    top_k: int = 6,
    doc_filter=None,
    user_request_id: str = None,
) -> RetrievalResult:
    raw_chunks = execute_tool(
        "retrieve_documents",
        stats,
        run_id=run_id,
        user_request_id=user_request_id,
        query=query,
        top_k=top_k,
        doc_filter=doc_filter,
    )

    clean_chunks = []
    flagged_count = 0
    for chunk in raw_chunks:
        decision = guardrail.scan_chunk(chunk["text"], chunk["doc_name"])
        if decision.blocked:
            flagged_count += 1
            stats.record_injection_attempt()
            stats.record_block()
            log_event(
                run_id=run_id,
                agent_name="retrieval_agent",
                action="CHUNK_DROPPED",
                user_request_id=user_request_id,
                security_decision=decision.action,
                security_reason=decision.reason,
                success=True,
                extra={"doc_name": chunk["doc_name"], "chunk_index": chunk["chunk_index"]},
            )
            continue
        if decision.action == "REVIEW":
            log_event(
                run_id=run_id,
                agent_name="retrieval_agent",
                action="CHUNK_FLAGGED",
                user_request_id=user_request_id,
                security_decision=decision.action,
                security_reason=decision.reason,
                success=True,
                extra={"doc_name": chunk["doc_name"], "chunk_index": chunk["chunk_index"]},
            )
        clean_chunks.append(chunk)

    return RetrievalResult(chunks=clean_chunks, flagged_count=flagged_count)
