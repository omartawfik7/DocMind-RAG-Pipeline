# DocMind — RAG Document Intelligence Pipeline

Ask questions about any document and get answers grounded in the source material — with citations, relevance scores, and full conversation history.

---

## What It Does

Upload a PDF, TXT, or Markdown file. Ask anything about it. DocMind retrieves the most relevant passages from your document and uses an LLM to generate a precise, cited answer — it never makes things up outside of what's in the file.

---

## Pipeline Architecture

```
Upload PDF / TXT / MD
        ↓
Extract raw text (PyPDF2)
        ↓
Chunk into 512-char segments with 80-char overlap
        ↓
Embed each chunk (sentence-transformers: all-MiniLM-L6-v2, runs locally)
        ↓
Store vectors in ChromaDB (persistent, on-disk)
        ↓
User asks a question
        ↓
Embed query → cosine similarity search → retrieve top 6 chunks
        ↓
Send chunks + question to Groq (llama-3.3-70b-versatile)
        ↓
Answer + source citations returned to UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | sentence-transformers `all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB (persistent, local) |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Backend | Python 3.11 + Flask |
| Frontend | Vanilla HTML / CSS / JS |

---

## Setup

### 1. Requirements
Python 3.11 is required. ChromaDB and several other dependencies do not support Python 3.12+ yet.

Download Python 3.11: https://www.python.org/downloads/release/python-3119/

### 2. Install dependencies
```bash
py -3.11 -m pip install -r requirements.txt
```

### 3. Get a free Groq API key
1. Go to https://console.groq.com/keys
2. Sign up (free, no credit card required)
3. Click **Create API Key** and copy it

### 4. Add your key to rag_engine.py
Open `rag_engine.py` and find this line:
```python
_groq_client = Groq(api_key="PASTE_YOUR_GROQ_KEY_HERE")
```
Replace `PASTE_YOUR_GROQ_KEY_HERE` with your actual key.

### 5. Run
```bash
py -3.11 app.py
```
Open **http://localhost:5000**

---

## Features

- **Any document type** — PDF, TXT, or Markdown
- **Drag-and-drop upload** with real-time ingestion progress
- **Chat interface** with full multi-turn conversation history
- **Source citations** on every answer with relevance % scores
- **Expandable excerpts** — click any citation to see the full passage
- **Document filtering** — chat with one document or search across all
- **Duplicate detection** — same file won't be re-indexed twice
- **Persistent vector store** — documents survive app restarts
- **Delete documents** from the index at any time

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/api/health` | Model status + document count |
| `POST` | `/api/upload` | Upload and ingest a document |
| `POST` | `/api/chat` | Ask a question, receive answer + citations |
| `GET` | `/api/documents` | List all ingested documents |
| `DELETE` | `/api/documents/<name>` | Remove a document from the index |

---

## Project Structure

```
rag_pipeline/
├── app.py              # Flask API and route handlers
├── rag_engine.py       # Core pipeline: chunking, embedding, retrieval, generation
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Chat UI
├── uploads/            # Uploaded files (auto-created)
└── chroma_db/          # Vector store (auto-created, persistent)
```

---

## Key Design Decisions

**Why sentence-transformers for embedding?**
Runs entirely locally — no API cost, no data leaves your machine during ingestion. `all-MiniLM-L6-v2` is fast, lightweight, and performs well on semantic search tasks.

**Why ChromaDB?**
Persistent local vector database with no infrastructure setup. Documents survive app restarts and don't need to be re-ingested.

**Why chunk with overlap?**
An 80-character overlap between chunks ensures that sentences split across chunk boundaries are still retrievable — prevents information loss at the edges.

**Why Groq?**
Free tier with fast inference. `llama-3.3-70b-versatile` provides near GPT-4 quality responses for document Q&A tasks.
