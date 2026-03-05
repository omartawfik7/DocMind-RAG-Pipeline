# DocMind — RAG Document Intelligence Pipeline

Ask questions about any PDF, TXT, or Markdown file. Answers are grounded in your documents with source citations and relevance scores.

## Stack
- **Embedding:** sentence-transformers `all-MiniLM-L6-v2` (local, free)
- **Vector Store:** ChromaDB (persistent, local)
- **LLM:** OpenAI GPT-4o
- **Backend:** Python + Flask
- **Frontend:** Vanilla HTML/CSS/JS

## Pipeline
```
Upload PDF/TXT → Extract text → Chunk (512 chars, 80 overlap)
→ Embed (MiniLM) → Store in ChromaDB
→ User question → Embed query → Cosine similarity search
→ Top-6 chunks → GPT-4o with context → Answer + citations
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
# Windows
set OPENAI_API_KEY=sk-...

# Mac/Linux
export OPENAI_API_KEY=sk-...
```
Get your key at: https://platform.openai.com/api-keys

### 3. Run
```bash
python app.py
```
Open **http://localhost:5000**

## Features
- Upload any PDF, TXT, or Markdown file
- Drag-and-drop upload with ingestion progress
- Chat with full conversation history (last 6 turns)
- Filter chat to a single document or search all at once
- Source citations with relevance scores on every answer
- Click citations to expand full excerpt
- Duplicate detection (same file won't be re-indexed)
- Delete documents from the index
