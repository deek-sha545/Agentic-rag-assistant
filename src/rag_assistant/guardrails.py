"""Guardrails against prompt injection hidden inside documents.

Threat: anyone who can get text into the document collection can write
"Ignore all previous instructions..." and hope the LLM obeys it. We defend in
three layers (defence in depth – no single layer is perfect):

  1. Detection + redaction (this file): instruction-like sentences and hidden
     HTML comments are replaced by a placeholder before the LLM sees them.
     Flagged chunks are also marked at ingestion time.
  2. Prompt hardening (prompts.py): passages are wrapped in <passage> tags and
     the system prompt says their content is *data*, never instructions.
  3. Output checks (this file + the grounding step in agent.py): answers that
     contain links not present in the sources are flagged, and every answer is
     checked against the retrieved passages.

Layer 1 is a simple keyword/regex approach. It is cheap and explainable, but a
determined attacker can rephrase around it - which is why layers 2 and 3 exist
and why the tests also run with layer 1 switched off.
"""

from __future__ import annotations

import re

REDACTION = "[removed: text that looked like instructions to an AI]"

# Hidden content: HTML comments are invisible to human readers of a web page
# but fully visible to an LLM, a classic hiding place for injections.
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# Phrases that address the AI or try to change its behaviour. They are rare in
# encyclopedic text, so false positives are uncommon.
INJECTION_PATTERNS = [
    r"\bignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"\bdisregard (all |any |your )?(previous |prior )?(instructions|rules|guidelines)",
    r"\bforget (all |your )?(previous |prior )?(instructions|rules)",
    r"\byou are (now|no longer)\b",
    r"\bfrom now on,? (you|the assistant)\b",
    r"^\s*(system|assistant)\s*:",
    r"\bnew (policy|instructions?|rules?)\b.{0,40}\b(assistant|ai|model)\b",
    r"\b(note|message|instructions?) (to|for) the (ai|assistant|model|llm|chatbot)\b",
    r"\b(ai|assistant|chatbot|language model)\b.{0,30}\b(must|should|need to) (respond|answer|reply|append|say|print|reveal|tell)",
    r"\byou must (respond|answer|reply|append|say|print|reveal|always|never)\b",
    r"\b(reveal|print|show|repeat) (your|the) (full |entire )?(system )?(prompt|instructions)",
    r"\benter (your|their) password\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]

# Sentence splitter used for redaction: split after . ! ? or at a blank line.
# (A single line break is NOT a sentence end: an attack sentence wrapped over
# two lines must be redacted as a whole.)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
URL = re.compile(r"https?://[^\s)\]>\"']+")


def find_injections(text: str) -> list[str]:
    """Return the suspicious snippets found in `text` (empty list = looks clean)."""
    hits = [m.group(0) for m in HTML_COMMENT.finditer(text)]
    for pattern in _COMPILED:
        hits.extend(m.group(0) for m in pattern.finditer(text))
    return hits


def is_suspicious(text: str) -> bool:
    return bool(find_injections(text))


def sanitize(text: str) -> tuple[str, bool]:
    """Redact instruction-like sentences. Returns (clean_text, was_changed)."""
    text = _SINGLE_NEWLINE.sub(" ", text)
    cleaned = HTML_COMMENT.sub(REDACTION, text)
    kept: list[str] = []
    for sentence in _SENTENCES.split(cleaned):
        if not sentence.strip():
            continue
        if any(p.search(sentence) for p in _COMPILED):
            # Collapse consecutive redactions into one placeholder.
            if not kept or kept[-1] != REDACTION:
                kept.append(REDACTION)
        else:
            kept.append(sentence.strip())
    cleaned = " ".join(kept)
    return cleaned, cleaned != " ".join(s.strip() for s in _SENTENCES.split(text) if s.strip())


def unexpected_urls(answer: str, allowed_urls: set[str]) -> list[str]:
    """URLs in the answer other than the source documents' own URLs.

    Citations (with links) are added by our code, not written by the LLM, so
    the LLM has no reason to output a link at all. A link inside the answer
    text is a strong sign that it followed an injected instruction - even if
    the link appears somewhere inside a (poisoned) passage.
    """
    return [url for url in URL.findall(answer) if url.rstrip(".,;") not in allowed_urls]


def remove_urls(answer: str, urls: list[str]) -> str:
    """Strip the given URLs from an answer."""
    for url in urls:
        answer = answer.replace(url, "[link removed]")
    return answer
