"""Step 2 of ingestion: cut sections into overlapping chunks.

Why chunk at all? Embedding models work best on short passages, and the LLM
prompt has limited room. We split each *section* separately so a chunk never
mixes two sections - that keeps citations ("document › section") accurate.

Strategy (simple and predictable):
  1. Split the section text into sentences.
  2. Add sentences to the current chunk until it would exceed `chunk_size`
     characters.
  3. Start the next chunk with the last few sentences of the previous one
     (up to `chunk_overlap` characters) so facts on a boundary are not lost.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from rag_assistant.documents import Section

# A sentence ends with . ! or ? followed by whitespace; newlines also split.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Chunk:
    """A piece of text that gets embedded and stored in the vector database."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def header(self) -> str:
        """'Mars › Moons' – used in prompts and citations."""
        return f"{self.metadata['title']} › {self.metadata['section']}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s and s.strip()]


def _hard_wrap(sentence: str, size: int) -> list[str]:
    """Very long 'sentences' (tables, lists) are cut into fixed-size pieces."""
    return [sentence[i:i + size] for i in range(0, len(sentence), size)]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one text into chunks of roughly `chunk_size` characters."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences: list[str] = []
    for sentence in split_sentences(text):
        sentences.extend(_hard_wrap(sentence, chunk_size) if len(sentence) > chunk_size else [sentence])

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > chunk_size:
            chunks.append(" ".join(current))
            # Keep trailing sentences as overlap for the next chunk.
            overlap: list[str] = []
            for previous in reversed(current):
                if len(" ".join([previous] + overlap)) > chunk_overlap:
                    break
                overlap.insert(0, previous)
            current = overlap
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_sections(sections: list[Section], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Turn sections into Chunk objects with the metadata needed for citations."""
    chunks: list[Chunk] = []
    for section in sections:
        for index, piece in enumerate(chunk_text(section.text, chunk_size, chunk_overlap)):
            # A deterministic id: re-ingesting the same data gives the same ids.
            raw_id = f"{section.doc_id}|{section.section}|{index}|{piece[:50]}"
            chunk_id = hashlib.sha1(raw_id.encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=piece,
                metadata={
                    "doc_id": section.doc_id,
                    "title": section.title,
                    "section": section.section,
                    "url": section.url,
                    "source": section.source,
                    "chunk_index": index,
                },
            ))
    return chunks
