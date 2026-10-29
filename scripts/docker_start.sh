#!/bin/sh
# Container start-up: build the vector store on first run, then start the API.
set -e
if ! python -c "from rag_assistant.config import load_settings; from rag_assistant.vector_store import VectorStore; import sys; sys.exit(0 if VectorStore(load_settings()).count() else 1)"; then
    echo "Vector store empty - downloading articles and ingesting (first run only)..."
    python scripts/download_data.py
    python -m rag_assistant.ingest
fi
exec uvicorn rag_assistant.api:app --host 0.0.0.0 --port 8000
