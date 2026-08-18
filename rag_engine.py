"""
rag_engine.py — Core RAG Pipeline
===================================
Document ingestion -> chunking -> embedding -> vector store -> retrieval

Pipeline:
  1. PDF/TXT ingestion via PyPDF2
  2. Recursive text chunking with overlap
  3. fastembed (ONNX Runtime) embedding -- all-MiniLM-L6-v2, no PyTorch dependency
  4. Qdrant Cloud vector store (persistent, hosted -- free tier)
  5. Cosine similarity retrieval

Answer generation now lives in agents/analysis_agent.py, behind the
llm_providers abstraction (Groq by default, or Anthropic/OpenAI via
LLM_PROVIDER) -- this module owns ingestion, chunking, and retrieval
only, so it has a single responsibility and eval/eval_retrieval.py's
existing `from rag_engine import ingest_document, retrieve` keeps
working unmodified.

Why Qdrant Cloud instead of local ChromaDB:
  Render's free web-service tier does not support persistent disks, so
  anything written to local disk (including a local ChromaDB store) is
  not guaranteed to survive a restart or redeploy. Qdrant Cloud's free
  tier is a real hosted database that persists independently of the app
  server, so ingested documents survive restarts, redeploys, and the
  free tier's spin-down/spin-up cycle.
"""

import os
import re
import uuid
import hashlib
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
from fastembed import TextEmbedding
import PyPDF2

# -- Config ------------------------------------------------------
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"  # served via fastembed (ONNX), not PyTorch -- keeps memory low on free-tier hosts
EMBED_DIM       = 384       # output dimension of all-MiniLM-L6-v2
CHUNK_SIZE      = 512       # characters per chunk
CHUNK_OVERLAP   = 80        # overlap between chunks
TOP_K           = 6         # number of chunks to retrieve
GROQ_MODEL      = "openai/gpt-oss-120b"  # default LLM_PROVIDER; see llm_providers.py.
                                          # NOTE: Groq retired llama-3.3-70b-versatile
                                          # (the model this repo originally shipped with)
                                          # since this app was last deployed; this is its
                                          # current replacement. Override with GROQ_MODEL.
COLLECTION_NAME = "documents"

# Secrets come from environment variables -- never hardcode API keys.
# Set QDRANT_URL and QDRANT_API_KEY in your host's environment (locally:
# a .env file loaded by python-dotenv; on Render: the service's
# Environment tab). The LLM provider's own key (GROQ_API_KEY /
# ANTHROPIC_API_KEY / OPENAI_API_KEY) is read lazily by llm_providers.py,
# only for whichever provider LLM_PROVIDER selects.
QDRANT_URL     = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

# -- Singleton client/model loaders -------------------------------
_embedder: Optional[TextEmbedding] = None
_qdrant_client: Optional[QdrantClient] = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        print("Loading embedding model (first run only)...")
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def get_collection() -> QdrantClient:
    """Returns a ready Qdrant client with the collection and required
    payload indexes created if needed.

    Explicitly connects over port 443 (standard HTTPS) rather than
    Qdrant's default gRPC-adjacent port 6333 -- some networks (school,
    corporate, certain ISPs/routers) block outbound traffic on
    non-standard ports like 6333 while leaving 443 wide open, since
    443 is required for essentially all web browsing to work at all.
    Qdrant Cloud's REST API is fully available over 443, so this avoids
    the problem without needing any network/firewall changes.

    Qdrant Cloud (unlike local/in-memory Qdrant) also requires an explicit
    payload index on any field used in a filter -- without it, filtered
    queries (like the duplicate-detection check on file_hash, or the
    doc_name filter used by retrieve/delete) return a 400 error.
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=443)
        existing = {c.name for c in _qdrant_client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        # Required for filtering on Qdrant Cloud -- safe to call even if
        # the index already exists (Qdrant no-ops on a duplicate request).
        for field in ("file_hash", "doc_name"):
            _qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
            )
    return _qdrant_client


def collection_count() -> int:
    client = get_collection()
    info = client.get_collection(COLLECTION_NAME)
    return info.points_count or 0


# ==================================================================
#  DOCUMENT INGESTION
# ==================================================================

def extract_text_from_pdf(filepath: str) -> str:
    """Extract full text from a PDF file."""
    text_parts = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
    return "\n\n".join(text_parts)


def extract_text_from_txt(filepath: str) -> str:
    """Extract text from a plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(filepath: str) -> str:
    """Route to correct extractor based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in [".txt", ".md"]:
        return extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ==================================================================
#  TEXT CHUNKING
# ==================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Recursive character-level chunking with overlap.
    Tries to split on paragraph breaks first, then sentence
    boundaries, then falls back to hard character limit.

    Returns list of dicts: {text, chunk_index, char_start, char_end}
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
        else:
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break
            else:
                sent_break = max(
                    text.rfind(". ", start, end),
                    text.rfind(".\n", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("? ", start, end),
                )
                if sent_break > start + chunk_size // 2:
                    end = sent_break + 1
            chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append({
                "text":        chunk,
                "chunk_index": chunk_idx,
                "char_start":  start,
                "char_end":    start + len(chunk),
            })
            chunk_idx += 1

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ==================================================================
#  EMBEDDING + VECTOR STORE
# ==================================================================

def ingest_document(filepath: str, doc_name: str) -> dict:
    """
    Full ingestion pipeline:
    extract -> chunk -> embed -> store in Qdrant Cloud

    Returns summary dict.
    """
    client   = get_collection()
    embedder = get_embedder()

    # Check if already ingested (by file hash)
    file_hash = hashlib.md5(open(filepath, "rb").read()).hexdigest()
    existing, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]),
        limit=1,
    )
    if existing:
        count, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]),
            limit=10_000,
        )
        return {
            "status":    "already_exists",
            "doc_name":  doc_name,
            "chunks":    len(count),
            "file_hash": file_hash,
        }

    print(f"  Extracting text from: {doc_name}")
    raw_text = extract_text(filepath)
    if not raw_text.strip():
        raise ValueError("No text could be extracted from this document.")

    print(f"  Chunking text ({len(raw_text):,} chars)...")
    chunks = chunk_text(raw_text)
    print(f"  Created {len(chunks)} chunks")

    print(f"  Embedding {len(chunks)} chunks...")
    texts      = [c["text"] for c in chunks]
    embeddings = [emb.tolist() for emb in embedder.embed(texts)]

    doc_id = str(uuid.uuid4())[:8]
    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "text":        chunk["text"],
                "doc_name":    doc_name,
                "doc_id":      doc_id,
                "file_hash":   file_hash,
                "chunk_index": chunk["chunk_index"],
                "char_start":  chunk["char_start"],
                "char_end":    chunk["char_end"],
            },
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Stored {len(chunks)} chunks for '{doc_name}'")

    return {
        "status":    "ingested",
        "doc_name":  doc_name,
        "doc_id":    doc_id,
        "chunks":    len(chunks),
        "chars":     len(raw_text),
        "file_hash": file_hash,
    }


# ==================================================================
#  RETRIEVAL
# ==================================================================

def retrieve(query: str, top_k: int = TOP_K,
             doc_filter: Optional[str] = None) -> list[dict]:
    """
    Embed query -> cosine similarity search -> return top-k chunks
    with metadata.
    """
    client   = get_collection()
    embedder = get_embedder()

    query_embedding = next(embedder.embed([query])).tolist()

    qfilter = None
    if doc_filter:
        qfilter = Filter(must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_filter))])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=qfilter,
        limit=top_k,
    ).points

    chunks = []
    for r in results:
        chunks.append({
            "text":        r.payload["text"],
            "doc_name":    r.payload["doc_name"],
            "chunk_index": r.payload["chunk_index"],
            "score":       round(r.score, 4),
        })

    return chunks


# ==================================================================
#  DOCUMENT MANAGEMENT
# ==================================================================

def list_documents() -> list[dict]:
    """Return all unique documents in the vector store."""
    client = get_collection()
    all_points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=10_000, with_payload=True)
    if not all_points:
        return []

    seen, docs = set(), []
    counts = {}
    for p in all_points:
        doc_name = p.payload["doc_name"]
        counts[doc_name] = counts.get(doc_name, 0) + 1
        if doc_name not in seen:
            seen.add(doc_name)
            docs.append({"doc_name": doc_name, "doc_id": p.payload["doc_id"]})

    for doc in docs:
        doc["chunk_count"] = counts[doc["doc_name"]]

    return docs


def delete_document(doc_name: str) -> dict:
    """Remove all chunks for a document from the vector store."""
    client = get_collection()
    matches, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]),
        limit=10_000,
    )
    if not matches:
        return {"status": "not_found", "doc_name": doc_name}

    ids = [p.id for p in matches]
    client.delete(collection_name=COLLECTION_NAME, points_selector=ids)
    return {"status": "deleted", "doc_name": doc_name, "chunks_removed": len(ids)}
