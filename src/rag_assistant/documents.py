"""Step 1 of ingestion: load documents and split them into sections.

Two input formats are supported:
  * data/raw/*.json  – Wikipedia articles from scripts/download_data.py, where
    headings look like "== History ==" (more "=" signs = deeper level).
  * *.md             – Markdown files (the poisoned test documents), where
    headings look like "## History".

Every document becomes a list of `Section` objects. Keeping the section name is
what lets the agent cite "Mars › Moons" instead of just "Mars".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Wikipedia sections that contain lists of links rather than prose.
SKIPPED_SECTIONS = {
    "see also", "references", "external links", "notes", "further reading",
    "bibliography", "citations", "sources", "footnotes", "explanatory notes",
}

WIKI_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")   # == Title ==
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")            # ## Title


@dataclass
class Section:
    """One section of one document."""

    doc_id: str      # stable id, e.g. "mars"
    title: str       # document title, e.g. "Mars"
    section: str     # heading path, e.g. "Moons" or "Geology > Volcanoes"
    text: str        # plain text of this section (without the heading)
    url: str = ""    # where the document came from
    source: str = "wikipedia"   # "wikipedia" or "local"


def split_into_sections(text: str, heading_pattern: re.Pattern, base_level: int) -> list[tuple[str, str]]:
    """Split text into (section_path, body) pairs using a heading regex.

    `base_level` is the number of markers ("=" or "#") of a top-level section.
    Text before the first heading is called "Introduction".
    """
    sections: list[tuple[str, str]] = []
    path: list[str] = []          # current heading path, e.g. ["Geology", "Volcanoes"]
    current_name = "Introduction"
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_name, body))

    for line in text.splitlines():
        match = heading_pattern.match(line)
        if not match:
            buffer.append(line)
            continue
        flush()
        buffer = []
        depth = max(len(match.group(1)) - base_level, 0)   # 0 = top level
        path = path[:depth] + [match.group(2).strip()]
        current_name = " > ".join(path)
    flush()

    # Drop reference-list style sections ("See also", "References" ...).
    return [(name, body) for name, body in sections
            if name.split(" > ")[0].lower() not in SKIPPED_SECTIONS]


def load_wikipedia_json(path: Path) -> list[Section]:
    """Load one article saved by scripts/download_data.py."""
    article = json.loads(path.read_text())
    return [
        Section(doc_id=path.stem, title=article["title"], section=name, text=body,
                url=article["url"], source="wikipedia")
        for name, body in split_into_sections(article["text"], WIKI_HEADING, base_level=2)
    ]


def load_markdown(path: Path) -> list[Section]:
    """Load a Markdown file. The first "# Title" line becomes the document title."""
    text = path.read_text()
    title = path.stem.replace("_", " ").title()
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if first_line.startswith("# "):
        title = first_line[2:].strip()
        text = text.lstrip()[len(first_line):]
    return [
        Section(doc_id=path.stem, title=title, section=name, text=body,
                url=f"local://{path.name}", source="local")
        for name, body in split_into_sections(text, MD_HEADING, base_level=2)
    ]


def load_directory(directory: Path) -> list[Section]:
    """Load every supported file in a folder (README.md files are skipped)."""
    sections: list[Section] = []
    for path in sorted(Path(directory).glob("*")):
        if path.suffix == ".json":
            sections.extend(load_wikipedia_json(path))
        elif path.suffix == ".md" and path.name.lower() != "readme.md":
            sections.extend(load_markdown(path))
    return sections
