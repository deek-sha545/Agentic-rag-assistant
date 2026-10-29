"""Tests for the scoring and metric code (synthetic results, fake judge)."""

import math

from conftest import FakeChatModel

from rag_assistant.config import Settings
from rag_assistant.evaluation import score_question, summarize
from rag_assistant.llm import LLMClient

ANSWERABLE = {"id": "a1", "question": "Moons of Mars?", "reference_answer": "Phobos and Deimos.",
              "answerable": True, "gold_doc_ids": ["mars"]}
UNANSWERABLE = {"id": "u1", "question": "Capital of Brazil?", "reference_answer": "Not in the documents.",
                "answerable": False, "gold_doc_ids": []}


def result(answer="Phobos and Deimos [1].", refused=False, cited=("mars",), retrieved=("mars",)):
    return {"answer": answer, "refused": refused,
            "citations": [{"doc_id": d} for d in cited], "retrieved": [{"doc_id": d} for d in retrieved],
            "retries": 0, "queries": ["q"],
            "usage": {"latency_s": 1.0, "llm_calls": 3, "input_tokens": 100, "output_tokens": 10,
                      "cost_usd": 0.001}}


def judge_saying(verdict):
    model = FakeChatModel(lambda s, u: f'{{"verdict": "{verdict}", "reason": "test"}}')
    return LLMClient(Settings(), provider="fake", model="judge", chat_model=model)


def test_correct_answer_with_right_citation():
    record = score_question(ANSWERABLE, result(), judge_saying("correct"))
    assert record["score"] == 1.0 and record["citation_hit"] and record["citation_precision"] == 1.0
    assert not record["hallucinated"]


def test_wrong_answer_counts_as_hallucination():
    record = score_question(ANSWERABLE, result(cited=("jupiter",)), judge_saying("incorrect"))
    assert record["hallucinated"] and record["citation_hit"] is False


def test_refusals_are_scored_without_judge():
    assert score_question(UNANSWERABLE, result(refused=True, cited=()), None)["score"] == 1.0
    assert score_question(ANSWERABLE, result(refused=True, cited=()), None)["score"] == 0.0


def test_summary_metrics():
    records = [
        score_question(ANSWERABLE, result(), judge_saying("correct")),
        score_question(ANSWERABLE | {"id": "a2"}, result(refused=True, cited=()), None),
        score_question(UNANSWERABLE, result(refused=True, cited=()), None),
        score_question(UNANSWERABLE | {"id": "u2"}, result(cited=()), judge_saying("incorrect")),
    ]
    s = summarize(records)
    assert s["answer_accuracy"] == 0.5
    assert s["false_refusal_rate"] == 0.5
    assert s["correct_refusal_rate"] == 0.5
    assert s["hallucination_rate"] == 0.25
    assert s["citation_hit_rate"] == 1.0
    assert not math.isnan(s["latency_mean_s"])
