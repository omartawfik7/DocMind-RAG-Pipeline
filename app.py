"""
app.py — Flask API for RAG Pipeline
=====================================
Endpoints:
  POST /api/upload          — Upload and ingest a PDF/TXT
  POST /api/chat            — Ask a question, get answer + citations
  GET  /api/documents       — List all ingested documents
  DELETE /api/documents/<name> — Remove a document
  GET  /api/health          — Health check
  GET  /                    — Dashboard UI
"""

import os
import traceback
from pathlib import Path
from flask import Flask, jsonify, request, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from rag_engine import (
    ingest_document, retrieve, generate_answer,
    list_documents, delete_document, get_collection
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "rag-dev-secret-change-in-prod")
CORS(app)

UPLOAD_FOLDER   = "./uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        collection = get_collection()
        doc_count  = len(list_documents())
        chunk_count = collection.count()
        return jsonify({
            "status":      "ok",
            "documents":   doc_count,
            "chunks":      chunk_count,
            "embed_model": "all-MiniLM-L6-v2",
            "llm":         "gpt-4o",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported. Use PDF, TXT, or MD."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        doc_name = Path(filename).stem  # filename without extension
        result   = ingest_document(filepath, doc_name)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data  = request.get_json(force=True) or {}
    query = str(data.get("query", "")).strip()
    doc_filter   = data.get("doc_filter") or None
    chat_history = data.get("history", [])

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Retrieve relevant chunks
        chunks = retrieve(query, doc_filter=doc_filter)
        if not chunks:
            return jsonify({
                "status": "ok",
                "answer": "No documents have been uploaded yet. Please upload a PDF or text file first.",
                "sources": [],
                "chunks_retrieved": 0,
            })

        # Generate answer
        result = generate_answer(query, chunks, chat_history)
        return jsonify({
            "status":           "ok",
            "answer":           result["answer"],
            "sources":          result["sources"],
            "chunks_retrieved": len(chunks),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/api/documents", methods=["GET"])
def documents():
    try:
        docs = list_documents()
        return jsonify({"status": "ok", "documents": docs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/documents/<doc_name>", methods=["DELETE"])
def delete_doc(doc_name: str):
    try:
        result = delete_document(doc_name)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("\n  RAG Document Chat — Flask API")
    print("  ==============================")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
