"""Evaluation: run the agent over eval/questions.jsonl and score every answer.

Metrics (see docs/DECISIONS.md for why these):
  answer_accuracy      mean judge score on answerable questions
                       (correct = 1, partially correct = 0.5, wrong or refused = 0)
  correct_refusal_rate share of unanswerable questions the agent declined
                       (or whose false premise it correctly rejected)
  false_refusal_rate   share of answerable questions the agent wrongly declined
  hallucination_rate   share of ALL questions answered with wrong/unsupported content
  citation_hit_rate    answered answerable questions citing the right document
  citation_precision   share of cited documents that are a right document
  retrieval_recall     answerable questions where the right document was among
                       the passages the answer was based on
  latency / tokens / cost per question

The judge is a separate (ideally stronger) LLM that compares the agent's
answer with the reference answer. It is kept the same for every experiment so
numbers are comparable.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from rag_assistant.agent import ResearchAssistant, parse_json_reply
from rag_assistant.config import PROJECT_ROOT, Settings
from rag_assistant.llm import LLMClient
from rag_assistant.memory import SessionMemory

logger = logging.getLogger(__name__)

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM = """\
You are a strict, fair grader for a question-answering system that must answer ONLY from
a document collection about the Solar System.

Compare the SYSTEM ANSWER with the REFERENCE ANSWER and output a verdict:
- "correct": contains the key facts of the reference and nothing that contradicts it.
  Extra correct detail is fine. Ignore citation markers like [1].
- "partially_correct": some key facts right, but incomplete or slightly inaccurate.
- "incorrect": wrong, contradicts the reference, or does not answer the question.

Special case - when the reference says the answer is NOT in the documents:
- "correct" if the system declines to answer, OR correctly points out a false premise
  (e.g. "Mars only has two moons") without inventing facts.
- "incorrect" if the system makes up an answer.

Reply with JSON only: {"verdict": "correct" | "partially_correct" | "incorrect", "reason": "<short>"}"""

JUDGE_USER = """\
QUESTION: {question}
REFERENCE ANSWER: {reference}
SYSTEM ANSWER: {answer}

JSON only:"""


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def judge(judge_llm: LLMClient, question: dict, answer: str) -> dict:
    """Ask the judge model for a verdict. Returns {"verdict", "score", "reason"}."""
    reply = judge_llm.complete(JUDGE_SYSTEM, JUDGE_USER.format(
        question=question["question"], reference=question["reference_answer"], answer=answer))
    parsed = parse_json_reply(reply) or {}
    verdict = parsed.get("verdict", "incorrect")
    if verdict not in SCORES:
        verdict = "incorrect"
    return {"verdict": verdict, "score": SCORES[verdict], "reason": str(parsed.get("reason", ""))[:300]}


def score_question(question: dict, result: dict, judge_llm: LLMClient | None) -> dict:
    """Combine the agent result with judge verdict and citation checks."""
    gold = set(question["gold_doc_ids"])
    cited = {c["doc_id"] for c in result["citations"]}
    retrieved = {p["doc_id"] for p in result["retrieved"]}
    refused = result["refused"]

    if refused:
        # A refusal needs no judge: right for unanswerable, wrong for answerable.
        verdict = {"verdict": "refused", "score": 0.0 if question["answerable"] else 1.0, "reason": ""}
    elif judge_llm is None:
        verdict = {"verdict": "not_judged", "score": float("nan"), "reason": ""}
    else:
        verdict = judge(judge_llm, question, result["answer"])

    record = {
        "id": question["id"],
        "question": question["question"],
        "answerable": question["answerable"],
        "answer": result["answer"],
        "refused": refused,
        "verdict": verdict["verdict"],
        "score": verdict["score"],
        "judge_reason": verdict["reason"],
        "gold_doc_ids": sorted(gold),
        "cited_doc_ids": sorted(cited),
        "citation_hit": bool(cited & gold) if gold and not refused else None,
        "citation_precision": (len(cited & gold) / len(cited)) if gold and cited else None,
        "retrieval_hit": bool(retrieved & gold) if gold else None,
        "retries": result["retries"],
        "queries": result["queries"],
        "latency_s": result["usage"]["latency_s"],
        "llm_calls": result["usage"]["llm_calls"],
        "tokens": result["usage"]["input_tokens"] + result["usage"]["output_tokens"],
        "cost_usd": result["usage"]["cost_usd"],
    }
    # Hallucination = a substantive answer that is wrong / made up.
    record["hallucinated"] = (not refused) and verdict["verdict"] == "incorrect"
    return record


def summarize(records: list[dict]) -> dict:
    """Aggregate per-question records into the headline metrics."""
    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.fmean(values), 3) if values else None

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    answered_ok = [r for r in answerable if not r["refused"]]
    latencies = sorted(r["latency_s"] for r in records)
    return {
        "questions": len(records),
        "answer_accuracy": mean([r["score"] for r in answerable]),
        "correct_refusal_rate": mean([r["score"] for r in unanswerable]),
        "false_refusal_rate": mean([float(r["refused"]) for r in answerable]),
        "hallucination_rate": mean([float(r["hallucinated"]) for r in records]),
        "citation_hit_rate": mean([float(r["citation_hit"]) for r in answered_ok if r["citation_hit"] is not None]),
        "citation_precision": mean([r["citation_precision"] for r in answered_ok]),
        "retrieval_recall": mean([float(r["retrieval_hit"]) for r in answerable]),
        "avg_retries": mean([r["retries"] for r in records]),
        "avg_llm_calls": mean([r["llm_calls"] for r in records]),
        "latency_mean_s": mean(latencies),
        "latency_p95_s": round(latencies[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
        "avg_tokens": mean([r["tokens"] for r in records]),
        "cost_per_question_usd": round(statistics.fmean(r["cost_usd"] for r in records), 6) if records else None,
    }


def run_experiment(name: str, settings: Settings, judge_llm: LLMClient | None,
                   questions: list[dict], out_dir: Path) -> dict:
    """Run every question once and save per-question records to <out_dir>/<name>.jsonl.

    Resumable: questions already in the file are skipped, so an interrupted run
    (e.g. hitting a rate limit) can simply be started again.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}.jsonl"
    done = {}
    if out_file.exists():
        done = {r["id"]: r for r in map(json.loads, out_file.read_text().splitlines())}

    agent = ResearchAssistant(settings, memory=SessionMemory(0))   # no memory: questions are independent
    with open(out_file, "a", encoding="utf-8") as f:
        for i, question in enumerate(questions, start=1):
            if question["id"] in done:
                continue
            result = agent.ask(question["question"], session_id=f"eval-{question['id']}").to_dict()
            record = score_question(question, result, judge_llm)
            done[question["id"]] = record
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            logger.info("[%s] %d/%d %s -> %s", name, i, len(questions), question["id"], record["verdict"])

    records = [done[q["id"]] for q in questions if q["id"] in done]
    return summarize(records)
