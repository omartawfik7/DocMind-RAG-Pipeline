"""
rag_engine.py — Core RAG Pipeline
===================================
Document ingestion → chunking → embedding → vector store → retrieval

Pipeline:
  1. PDF/TXT ingestion via PyPDF2
  2. Recursive text chunking with overlap
  3. Sentence-transformers embedding (all-MiniLM-L6-v2)
  4. ChromaDB vector store (persistent, local)
  5. Cosine similarity retrieval
  6. Claude API answer generation with citations
"""

import os
import re
import uuid
import json
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────
CHROMA_DIR      = "./chroma_db"
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 512       # characters per chunk
CHUNK_OVERLAP   = 80        # overlap between chunks
TOP_K           = 6         # number of chunks to retrieve
OPENAI_MODEL    = "gpt-4o"

# ── Singleton model loader ────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading embedding model (first run only)...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ══════════════════════════════════════════════════════════════
#  DOCUMENT INGESTION
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
#  TEXT CHUNKING
# ══════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Recursive character-level chunking with overlap.
    Tries to split on paragraph breaks first, then sentence
    boundaries, then falls back to hard character limit.

    Returns list of dicts: {text, chunk_index, char_start, char_end}
    """
    # Normalize whitespace
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
            # Try to break at paragraph
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break
            else:
                # Try sentence boundary
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


# ══════════════════════════════════════════════════════════════
#  EMBEDDING + VECTOR STORE
# ══════════════════════════════════════════════════════════════

def ingest_document(filepath: str, doc_name: str) -> dict:
    """
    Full ingestion pipeline:
    extract → chunk → embed → store in ChromaDB

    Returns summary dict.
    """
    collection = get_collection()
    embedder   = get_embedder()

    # Check if already ingested (by file hash)
    file_hash = hashlib.md5(open(filepath, "rb").read()).hexdigest()
    existing  = collection.get(where={"file_hash": file_hash})
    if existing["ids"]:
        return {
            "status":      "already_exists",
            "doc_name":    doc_name,
            "chunks":      len(existing["ids"]),
            "file_hash":   file_hash,
        }

    # Extract text
    print(f"  Extracting text from: {doc_name}")
    raw_text = extract_text(filepath)
    if not raw_text.strip():
        raise ValueError("No text could be extracted from this document.")

    # Chunk
    print(f"  Chunking text ({len(raw_text):,} chars)...")
    chunks = chunk_text(raw_text)
    print(f"  Created {len(chunks)} chunks")

    # Embed
    print(f"  Embedding {len(chunks)} chunks...")
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # Store in ChromaDB
    doc_id = str(uuid.uuid4())[:8]
    ids, docs, metas, embeds = [], [], [], []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}_chunk_{i}"
        ids.append(chunk_id)
        docs.append(chunk["text"])
        embeds.append(emb)
        metas.append({
            "doc_name":    doc_name,
            "doc_id":      doc_id,
            "file_hash":   file_hash,
            "chunk_index": chunk["chunk_index"],
            "char_start":  chunk["char_start"],
            "char_end":    chunk["char_end"],
        })

    collection.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
    print(f"  Stored {len(chunks)} chunks for '{doc_name}'")

    return {
        "status":    "ingested",
        "doc_name":  doc_name,
        "doc_id":    doc_id,
        "chunks":    len(chunks),
        "chars":     len(raw_text),
        "file_hash": file_hash,
    }


# ══════════════════════════════════════════════════════════════
#  RETRIEVAL
# ══════════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = TOP_K,
             doc_filter: Optional[str] = None) -> list[dict]:
    """
    Embed query → cosine similarity search → return top-k chunks
    with metadata.
    """
    collection = get_collection()
    embedder   = get_embedder()

    query_embedding = embedder.encode([query]).tolist()

    where = {"doc_name": doc_filter} if doc_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text":        results["documents"][0][i],
            "doc_name":    results["metadatas"][0][i]["doc_name"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "score":       round(1 - results["distances"][0][i], 4),
        })

    return chunks


# ══════════════════════════════════════════════════════════════
#  ANSWER GENERATION (Claude API)
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise document analyst. You answer questions strictly based on the provided document excerpts.

Rules:
- Answer only from the provided context. Do not use outside knowledge.
- If the context doesn't contain enough information, say so clearly.
- Always cite which document and chunk your answer comes from using [Source: doc_name, Chunk N].
- Be concise but complete. Use bullet points for multi-part answers.
- If quoting directly, use quotation marks and cite immediately after.
"""

def generate_answer(query: str, chunks: list[dict],
                    chat_history: list[dict] = None) -> dict:
    """
    Build context from retrieved chunks → call GPT-4o → return answer + sources.
    """
    if not chunks:
        return {
            "answer":  "No relevant content found in the uploaded documents.",
            "sources": [],
        }

    # Build context block
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Excerpt {i+1} | Source: {chunk['doc_name']}, Chunk {chunk['chunk_index']} | Relevance: {chunk['score']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""Document excerpts:

{context}

---

Question: {query}"""

    # Build messages with optional history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for turn in chat_history[-6:]:  # keep last 6 turns
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    client   = OpenAI()
    response = client.chat.completions.create(
        model      = OPENAI_MODEL,
        max_tokens = 1024,
        messages   = messages,
    )

    answer = response.choices[0].message.content

    # Extract unique sources cited
    sources = []
    seen    = set()
    for chunk in chunks:
        key = f"{chunk['doc_name']}::{chunk['chunk_index']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_name":    chunk["doc_name"],
                "chunk_index": chunk["chunk_index"],
                "score":       chunk["score"],
                "excerpt":     chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
            })

    return {"answer": answer, "sources": sources}


# ══════════════════════════════════════════════════════════════
#  DOCUMENT MANAGEMENT
# ══════════════════════════════════════════════════════════════

def list_documents() -> list[dict]:
    """Return all unique documents in the vector store."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_items = collection.get(include=["metadatas"])
    seen, docs = set(), []
    for meta in all_items["metadatas"]:
        doc_name = meta["doc_name"]
        if doc_name not in seen:
            seen.add(doc_name)
            docs.append({"doc_name": doc_name, "doc_id": meta["doc_id"]})

    # Count chunks per doc
    for doc in docs:
        result = collection.get(where={"doc_name": doc["doc_name"]})
        doc["chunk_count"] = len(result["ids"])

    return docs


def delete_document(doc_name: str) -> dict:
    """Remove all chunks for a document from the vector store."""
    collection = get_collection()
    results    = collection.get(where={"doc_name": doc_name})
    if not results["ids"]:
        return {"status": "not_found", "doc_name": doc_name}

    collection.delete(ids=results["ids"])
    return {"status": "deleted", "doc_name": doc_name, "chunks_removed": len(results["ids"])}
