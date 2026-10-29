"""The ingestion pipeline: load -> chunk -> scan for injections -> embed -> store.

Run from the command line (after scripts/download_data.py):
    python -m rag_assistant.ingest                 # chunk size from config.yaml
    python -m rag_assistant.ingest --chunk-sizes 500 1000 2000
"""

from __future__ import annotations

import argparse
import logging
import time

from rag_assistant.chunking import chunk_sections
from rag_assistant.config import Settings, load_settings
from rag_assistant.documents import load_directory
from rag_assistant.guardrails import is_suspicious
from rag_assistant.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest(settings: Settings, chunk_size: int | None = None, reset: bool = True) -> dict:
    """Build (or rebuild) the vector store for one chunk size. Returns statistics."""
    chunk_size = chunk_size or settings.chunk_size
    # Keep the overlap at the same *proportion* of the chunk (150/1000 = 15%).
    overlap = round(settings.chunk_overlap * chunk_size / settings.chunk_size)
    started = time.perf_counter()

    # 1. Load: real articles + the poisoned test documents.
    sections = load_directory(settings.path(settings.raw_data_dir))
    if not sections:
        raise SystemExit("No documents found. Run `python scripts/download_data.py` first.")
    sections += load_directory(settings.path(settings.poisoned_data_dir))

    # 2. Chunk.
    chunks = chunk_sections(sections, chunk_size, overlap)

    # 3. Guardrail scan: mark chunks that contain instruction-like text.
    flagged = 0
    for chunk in chunks:
        chunk.metadata["suspicious"] = is_suspicious(chunk.text)
        flagged += chunk.metadata["suspicious"]

    # 4. Embed + store.
    store = VectorStore(settings, chunk_size)
    if reset:
        store.reset()
    store.add_chunks(chunks)

    stats = {
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "documents": len({s.doc_id for s in sections}),
        "sections": len(sections),
        "chunks": len(chunks),
        "flagged_chunks": flagged,
        "seconds": round(time.perf_counter() - started, 1),
    }
    logger.info("Ingestion finished: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ChromaDB vector store.")
    parser.add_argument("--chunk-sizes", type=int, nargs="*",
                        help="One or more chunk sizes (characters). Default: from config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    for size in args.chunk_sizes or [settings.chunk_size]:
        print(ingest(settings, size))


if __name__ == "__main__":
    main()
