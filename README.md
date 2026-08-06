# DocMind: RAG Document Intelligence Pipeline

Ask questions about any document and get answers grounded in the source material, with citations, relevance scores, and full conversation history.

**Live demo:** https://docmind-rag-pipeline.onrender.com
(free tier hosting, so the first request after a period of inactivity may take 30 to 60 seconds to wake up)

---

## What It Does

Upload a PDF, TXT, or Markdown file. Ask anything about it. DocMind retrieves the most relevant passages from your document and uses an LLM to generate a precise, cited answer. It does not make things up outside of what is in the file, and every answer shows exactly which chunk of the document it came from.

---

## Pipeline Architecture

```
Upload PDF / TXT / MD
        v
Extract raw text (PyPDF2)
        v
Chunk into 512-char segments with 80-char overlap
        v
Embed each chunk (fastembed, ONNX Runtime, all-MiniLM-L6-v2)
        v
Store vectors in Qdrant Cloud (persistent, hosted)
        v
User asks a question
        v
Embed query, run cosine similarity search, retrieve top 6 chunks
        v
Send chunks and question to Groq (llama-3.3-70b-versatile)
        v
Answer and source citations returned to UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | fastembed (ONNX Runtime), `all-MiniLM-L6-v2` |
| Vector Store | Qdrant Cloud (persistent, hosted) |
| LLM | Groq API, `llama-3.3-70b-versatile` (free tier) |
| Backend | Python 3.11, Flask, gunicorn |
| Frontend | Vanilla HTML, CSS, JS |
| Hosting | Render (free tier) |

**Why fastembed instead of sentence-transformers:** sentence-transformers pulls in PyTorch as a dependency, and just importing PyTorch uses enough memory to exceed Render's free-tier 512MB limit before the app can even start. fastembed runs the same embedding model on ONNX Runtime instead, which uses a fraction of the memory and avoids the problem entirely.

**Why Qdrant Cloud instead of local storage:** Render's free web-service tier does not support persistent disks, so anything written to local disk is not guaranteed to survive a restart or redeploy. Qdrant Cloud's free tier is a real hosted database that persists independently of the app server.

**Why port 443:** Qdrant's client defaults to port 6333. Some networks (schools, offices, certain routers or ISPs) block outbound traffic on non-standard ports while leaving standard HTTPS (443) open, since 443 is required for basic web browsing to work at all. The app explicitly connects over 443 so it works from more networks without any firewall changes needed.

---

## Setup

### 1. Requirements
Python 3.11 or later.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get free API keys

**Groq (LLM):**
1. Go to https://console.groq.com/keys
2. Sign up, no credit card required
3. Click Create API Key and copy it

**Qdrant Cloud (vector store):**
1. Go to https://cloud.qdrant.io
2. Sign up, no credit card required
3. Create a free cluster
4. Copy the cluster URL and generate an API key from the cluster dashboard

### 4. Configure environment variables
Copy `.env.example` to `.env` and fill in your real values:
```bash
cp .env.example .env
```
```
GROQ_API_KEY=your_groq_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_key
SECRET_KEY=any_random_string
FLASK_DEBUG=false
```
Secrets are never hardcoded in source. `.env` is excluded from version control by `.gitignore`.

### 5. Run locally
```bash
python app.py
```
Open http://localhost:5000

### 6. Deploy
See `DEPLOYMENT_GUIDE.md` for a full walkthrough of deploying to Render with Qdrant Cloud, including the memory and networking issues above and how they were solved.

---

## Retrieval Evaluation

`eval/eval_retrieval.py` ingests a reference document with known facts and runs a fixed set of 10 questions against the retrieval layer, reporting hit-rate at top-6 and mean reciprocal rank (MRR).

```bash
cd eval
python eval_retrieval.py
```

Most recent run:
- Hit-rate@6: 100%
- MRR: 0.95
- Average retrieval latency: ~107ms

This tests whether the embedding and vector-search layer surfaces the right source material, independent of what the LLM does with it afterward. If retrieval is wrong, no amount of prompt engineering fixes the final answer.

---

## Features

- Any document type: PDF, TXT, or Markdown
- Drag-and-drop upload with real-time ingestion progress
- Chat interface with full multi-turn conversation history
- Source citations on every answer with relevance scores
- Expandable excerpts: click any citation to see the full passage
- Document filtering: chat with one document or search across all
- Duplicate detection by content hash, so the same file is never re-indexed twice
- Persistent, hosted vector store: documents survive app restarts and redeploys
- Delete documents from the index at any time

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `GET` | `/api/health` | Model status and document count |
| `POST` | `/api/upload` | Upload and ingest a document |
| `POST` | `/api/chat` | Ask a question, receive answer and citations |
| `GET` | `/api/documents` | List all ingested documents |
| `DELETE` | `/api/documents/<name>` | Remove a document from the index |

---

## Project Structure

```
DocMind-RAG-Pipeline/
├── app.py                  Flask API and route handlers
├── rag_engine.py            Core pipeline: chunking, embedding, retrieval, generation
├── requirements.txt         Python dependencies
├── Procfile                 Process definition for deployment (gunicorn)
├── render.yaml               Render deployment blueprint
├── .env.example              Template for local environment variables
├── .gitignore
├── templates/
│   └── index.html            Chat UI
├── eval/
│   ├── eval_retrieval.py     Retrieval quality evaluation harness
│   └── sample_doc.md         Reference document used by the eval harness
└── DEPLOYMENT_GUIDE.md        Full deployment walkthrough
```

---

## Key Design Decisions

**Why chunk with overlap?**
An 80-character overlap between chunks means sentences split across chunk boundaries are still retrievable in full, preventing information loss at the edges.

**Why Groq for generation?**
Free tier with fast inference. `llama-3.3-70b-versatile` gives strong quality for document Q&A while staying within a genuinely free usage tier, no credit card required.

**Why measure retrieval separately from generation?**
A RAG system can fail in two independent places: retrieval can return the wrong chunks, or the LLM can misuse correct chunks. The eval harness isolates the first failure mode so it can be diagnosed and fixed without guessing which layer is responsible.
