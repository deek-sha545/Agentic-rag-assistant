"""Download the document collection: ~40 Wikipedia articles about the Solar System.

Wikipedia text is licensed under CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). We keep each article's URL
and revision id so every passage can be attributed.

Run:
    python scripts/download_data.py

Each article is saved as data/raw/<slug>.json with the fields:
    title, url, revision_id, license, text
The text is plain text in which section headings look like "== History ==".
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API users to send a descriptive User-Agent.
HEADERS = {"User-Agent": "agentic-rag-assistant/0.1 (educational portfolio project)"}

ARTICLES = [
    # The Sun and planets
    "Sun", "Mercury (planet)", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
    "Uranus", "Neptune",
    # Moons
    "Moon", "Io (moon)", "Europa (moon)", "Ganymede (moon)", "Callisto (moon)",
    "Titan (moon)", "Enceladus", "Triton (moon)", "Phobos (moon)",
    # Dwarf planets and small bodies
    "Pluto", "Ceres (dwarf planet)", "Eris (dwarf planet)", "Asteroid belt",
    "Kuiper belt", "Oort cloud", "Comet", "Halley's Comet",
    # Features and formation
    "Great Red Spot", "Rings of Saturn", "Formation and evolution of the Solar System",
    "Solar System",
    # Missions
    "Voyager 1", "Voyager 2", "Cassini–Huygens", "Juno (spacecraft)",
    "New Horizons", "Galileo (spacecraft)", "Apollo 11", "Perseverance (rover)",
    "Curiosity (rover)", "Parker Solar Probe", "Dawn (spacecraft)",
]


def slugify(title: str) -> str:
    """'Io (moon)' -> 'io_moon' (safe file name)."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def fetch_article(title: str) -> dict:
    """Fetch the plain-text content of one Wikipedia article."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts|info|revisions",
        "rvprop": "ids",
        "inprop": "url",
        "explaintext": 1,             # plain text instead of HTML
        "exsectionformat": "wiki",    # keep "== Heading ==" markers
        "redirects": 1,
        "titles": title,
    }
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    page = response.json()["query"]["pages"][0]
    if "missing" in page:
        raise ValueError(f"Article not found: {title}")
    return {
        "title": page["title"],
        "url": page["fullurl"],
        "revision_id": page["revisions"][0]["revid"],
        "license": "CC BY-SA 4.0",
        "text": page["extract"],
    }


def fetch_with_retries(title: str, attempts: int = 4) -> dict | None:
    """Wikipedia sometimes rate-limits (HTTP 429) or drops connections, so retry
    with a growing pause (2s, 4s, 8s ...) before giving up on an article."""
    for attempt in range(1, attempts + 1):
        try:
            return fetch_article(title)
        except Exception as error:
            print(f"  attempt {attempt} failed for {title}: {str(error)[:80]}")
            time.sleep(2 ** attempt)
    print(f"FAILED: {title}")
    return None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for title in ARTICLES:
        out_path = OUTPUT_DIR / f"{slugify(title)}.json"
        if out_path.exists():
            print(f"skip (already downloaded): {title}")
            continue
        article = fetch_with_retries(title)
        if article is None:
            failures.append(title)
            continue
        out_path.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"saved {title:45s} {len(article['text']):>7,} chars")
        time.sleep(1.0)  # be polite to Wikipedia's servers

    print(f"\nDone. {len(ARTICLES) - len(failures)} articles in {OUTPUT_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
