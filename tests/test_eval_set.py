"""Sanity checks on the evaluation set itself (a bad eval set gives misleading numbers)."""

import json

import pytest

from rag_assistant.config import PROJECT_ROOT
from rag_assistant.evaluation import load_questions

RAW = PROJECT_ROOT / "data" / "raw"


def test_eval_set_size_and_mix():
    questions = load_questions()
    assert len(questions) >= 40
    assert len({q["id"] for q in questions}) == len(questions)
    unanswerable = [q for q in questions if not q["answerable"]]
    assert 5 <= len(unanswerable) < len(questions) / 2


def test_every_reference_is_backed_by_its_document():
    """Each answerable question quotes evidence that really exists in the gold article."""
    if not any(RAW.glob("*.json")):
        pytest.skip("run scripts/download_data.py first")
    for q in load_questions():
        if q["answerable"]:
            texts = [json.loads((RAW / f"{d}.json").read_text())["text"] for d in q["gold_doc_ids"]]
            assert any(q["evidence"] in t for t in texts), q["id"]
