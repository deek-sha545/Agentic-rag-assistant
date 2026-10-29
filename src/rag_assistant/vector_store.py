"""Steps 3 and 4 of ingestion (embed + store) and the search used by the agent.

* Embeddings: an open-source sentence-transformers model turns text into a
  vector (a list of 384 numbers for all-MiniLM-L6-v2). Similar meaning ->
  vectors pointing in a similar direction.
* Storage: ChromaDB keeps the vectors on disk (folder `chroma_db/`) and finds
  the nearest ones to a query vector.

Each chunk size gets its own collection (e.g. "wiki_c1000"), so experiments
can compare chunk sizes without re-ingesting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag_assistant.chunking import Chunk
from rag_assistant.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    """A retrieved chunk plus its similarity score (higher = more similar)."""

    chunk_id: str
    text: str
    title: str
    section: str
    url: str
    doc_id: str
    score: float
    suspicious: bool = False

    def header(self) -> str:
        return f"{self.title} › {self.section}"


@lru_cache(maxsize=2)
def get_embedder(model_name: str):
    """Load the embedding model once and reuse it (loading takes a few seconds)."""
    from sentence_transformers import SentenceTransformer  # slow import, do it lazily

    logger.info("Loading embedding model %s", model_name)
    return SentenceTransformer(model_name, device="cpu")


def embed(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed a list of texts. Vectors are normalised so cosine similarity works."""
    model = get_embedder(model_name)
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=len(texts) > 500)
    return vectors.tolist()


def text_for_embedding(title: str, section: str, text: str) -> str:
    """Prefix the chunk with its title and section before embedding.

    A chunk from 'Io (moon) › Volcanism' often never says "Io"; adding the
    header lets a question about Io find it.
    """
    return f"{title} - {section}\n{text}"


class VectorStore:
    """Thin wrapper around one ChromaDB collection."""

    def __init__(self, settings: Settings, chunk_size: int | None = None):
        self.settings = settings
        self.chunk_size = chunk_size or settings.chunk_size
        self.collection_name = f"wiki_c{self.chunk_size}"
        self.client = _get_client(str(settings.path(settings.chroma_dir)))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine distance between vectors
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (used before re-ingesting)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 500) -> None:
        """Embed and store chunks in batches."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = embed(
                [text_for_embedding(c.metadata["title"], c.metadata["section"], c.text) for c in batch],
                self.settings.embedding_model,
            )
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=vectors,
            )
            logger.info("Stored %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks))

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        """Return the k chunks most similar to the query."""
        k = k or self.settings.top_k
        query_vector = embed([query], self.settings.embedding_model)
        result = self.collection.query(query_embeddings=query_vector, n_results=k)
        passages = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            passages.append(Passage(
                chunk_id=chunk_id, text=text, title=meta["title"], section=meta["section"],
                url=meta.get("url", ""), doc_id=meta["doc_id"],
                score=round(1.0 - distance, 4),          # cosine similarity
                suspicious=bool(meta.get("suspicious", False)),
            ))
        return passages


@lru_cache(maxsize=4)
def _get_client(path: str):
    """One ChromaDB client per folder (creating several for one folder is unsafe)."""
    return chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))
