"""
eval_retrieval.py — Retrieval quality evaluation for the DocMind RAG pipeline
================================================================================
Ingests a reference document with known facts, runs a fixed set of
question / expected-keyword pairs against the retrieval layer, and reports:

  - Hit-rate@K   : fraction of questions where a chunk containing the
                   expected keyword(s) appears in the top-K retrieved chunks
  - MRR          : mean reciprocal rank of the first correct chunk
                   (rewards correct answers appearing higher in the ranking)

This is a retrieval-quality eval, not an answer-quality eval — it tests
whether the embedding + vector-search layer surfaces the right source
material, independent of what the LLM then does with it. That separation
matters: if retrieval is broken, no amount of prompt engineering on the
generation side will fix wrong answers.

Run from the eval/ directory after setting environment variables:
    python eval_retrieval.py
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from rag_engine import ingest_document, retrieve  # noqa: E402

DOC_PATH = str(Path(__file__).resolve().parent / "sample_doc.md")
DOC_NAME = "northfield-remote-work-policy"
TOP_K = 6

# Golden Q&A set. `expected_keywords` are strings that should appear
# (case-insensitive) in at least one retrieved chunk if retrieval worked.
EVAL_SET = [
    {
        "question": "How long is the probationary period before someone can request remote work?",
        "expected_keywords": ["90-day", "probationary"],
    },
    {
        "question": "What equipment does the city provide for remote employees?",
        "expected_keywords": ["laptop", "$150", "stipend"],
    },
    {
        "question": "What internet speed is required for remote work?",
        "expected_keywords": ["25 Mbps"],
    },
    {
        "question": "What are the core hours remote employees must be available?",
        "expected_keywords": ["10:00 AM", "3:00 PM"],
    },
    {
        "question": "Can remote employees store city data on their personal laptop?",
        "expected_keywords": ["personal devices", "prohibited"],
    },
    {
        "question": "How often must managers check in with remote direct reports?",
        "expected_keywords": ["video check-in", "once", "week"],
    },
    {
        "question": "How much notice does the city give before revoking remote work eligibility?",
        "expected_keywords": ["14 days"],
    },
    {
        "question": "What is the mileage reimbursement rate for remote employee business travel?",
        "expected_keywords": ["$0.67", "mile"],
    },
    {
        "question": "Do union employees need anything extra for remote work arrangements?",
        "expected_keywords": ["collective bargaining", "union"],
    },
    {
        "question": "Is the internet stipend paid every month?",
        "expected_keywords": ["not provided", "ongoing", "initial"],
    },
]


def run_eval():
    print(f"Ingesting reference document: {DOC_NAME}")
    ingest_document(DOC_PATH, DOC_NAME)

    results = []
    for case in EVAL_SET:
        start = time.time()
        chunks = retrieve(case["question"], top_k=TOP_K, doc_filter=DOC_NAME)
        latency_ms = round((time.time() - start) * 1000, 1)

        rank = None
        for i, c in enumerate(chunks):
            text_lower = c["text"].lower()
            if any(kw.lower() in text_lower for kw in case["expected_keywords"]):
                rank = i + 1
                break

        results.append({
            "question":      case["question"],
            "hit":           rank is not None,
            "rank":          rank,
            "top_score":     chunks[0]["score"] if chunks else None,
            "latency_ms":    latency_ms,
        })

    hit_rate = sum(r["hit"] for r in results) / len(results)
    mrr = sum((1 / r["rank"] if r["rank"] else 0) for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    report = {
        "eval_set_size":       len(EVAL_SET),
        "top_k":               TOP_K,
        f"hit_rate_at_{TOP_K}": round(hit_rate, 3),
        "mrr":                 round(mrr, 3),
        "avg_retrieval_latency_ms": round(avg_latency, 1),
        "cases": results,
    }

    print(json.dumps(report, indent=2))

    out_path = Path(__file__).resolve().parent / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")

    return report


if __name__ == "__main__":
    run_eval()
