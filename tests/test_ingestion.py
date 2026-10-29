"""Tests for loading documents and chunking (no network, no LLM)."""

import json

from rag_assistant.chunking import chunk_sections, chunk_text
from rag_assistant.documents import load_directory, load_markdown, split_into_sections, WIKI_HEADING


def test_wiki_sections_are_split_with_heading_path():
    text = "Intro text.\n== Geology ==\nRocks.\n=== Volcanoes ===\nBig ones.\n== References ==\nx"
    sections = split_into_sections(text, WIKI_HEADING, base_level=2)
    assert sections == [
        ("Introduction", "Intro text."),
        ("Geology", "Rocks."),
        ("Geology > Volcanoes", "Big ones."),
    ]  # "References" is dropped


def test_load_wikipedia_json(tmp_path):
    article = {"title": "Mars", "url": "https://en.wikipedia.org/wiki/Mars", "revision_id": 1,
               "license": "CC BY-SA 4.0", "text": "Mars is red.\n== Moons ==\nPhobos and Deimos."}
    (tmp_path / "mars.json").write_text(json.dumps(article))
    sections = load_directory(tmp_path)
    assert [(s.title, s.section) for s in sections] == [("Mars", "Introduction"), ("Mars", "Moons")]
    assert sections[1].url.endswith("/Mars")


def test_markdown_title_and_readme_skipped(tmp_path):
    (tmp_path / "notes.md").write_text("# My Notes\n\n## Part one\nHello.\n")
    (tmp_path / "README.md").write_text("# Not a document")
    sections = load_directory(tmp_path)
    assert len(sections) == 1
    assert sections[0].title == "My Notes" and sections[0].section == "Part one"


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # Overlap: the start of each chunk repeats the end of the previous one.
    assert chunks[1].split(".")[0] in chunks[0]


def test_very_long_sentence_is_hard_wrapped():
    chunks = chunk_text("x" * 2500, chunk_size=1000, chunk_overlap=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks).count("x") >= 2500


def test_chunk_ids_are_deterministic(tmp_path):
    (tmp_path / "a.md").write_text("# A\n## S\n" + "Words here. " * 200)
    sections = load_markdown(tmp_path / "a.md")
    first = [c.chunk_id for c in chunk_sections(sections, 400, 50)]
    second = [c.chunk_id for c in chunk_sections(sections, 400, 50)]
    assert first == second and len(set(first)) == len(first)
