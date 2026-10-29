"""Retrieval-only experiment: which chunk size finds the right passages best?

This needs NO LLM (no API key, no cost), so it is the first experiment to run.
For every answerable eval question and every chunk size we search the vector
store and measure:

  doc_hit@k       the gold article is among the top-k passages
  evidence_hit@k  a passage contains the exact evidence sentence fragment
                  (stricter: the answer itself was retrieved)
  mrr             mean reciprocal rank of the first gold-article passage
  poisoned@k      share of questions whose top-k contains a poisoned document
  context_chars   characters the LLM would have to read (cost proxy)

Run:  python scripts/eval_retrieval.py      (after ingesting all chunk sizes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_assistant.config import load_settings  # noqa: E402
from rag_assistant.evaluation import load_questions  # noqa: E402
from rag_assistant.vector_store import VectorStore  # noqa: E402

CHUNK_SIZES = [500, 1000, 2000]
K_VALUES = [3, 5, 8, 10]
RESULTS = Path(__file__).resolve().parents[1] / "results"
POISONED_TITLES = {"Mars Survey Notes", "Jupiter Fact Sheet", "Saturn Mission Brief",
                   "Pluto Reclassification Summary"}


def normalise(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    settings = load_settings()
    questions = [q for q in load_questions() if q["answerable"]]
    rows = []
    for size in CHUNK_SIZES:
        store = VectorStore(settings, chunk_size=size)
        if store.count() == 0:
            print(f"Collection for chunk size {size} is empty - run "
                  f"`python -m rag_assistant.ingest --chunk-sizes {size}`")
            continue
        hits = {q["id"]: store.search(q["question"], max(K_VALUES)) for q in questions}
        for k in K_VALUES:
            doc_hit, evidence_hit, rr, poisoned, chars = [], [], [], [], []
            for q in questions:
                top = hits[q["id"]][:k]
                gold = set(q["gold_doc_ids"])
                ranks = [i for i, p in enumerate(top, start=1) if p.doc_id in gold]
                doc_hit.append(bool(ranks))
                rr.append(1 / ranks[0] if ranks else 0.0)
                evidence_hit.append(any(normalise(q["evidence"]) in normalise(p.text) for p in top))
                poisoned.append(any(p.title in POISONED_TITLES for p in top))
                chars.append(sum(len(p.text) for p in top))
            rows.append({
                "chunk_size": size, "k": k,
                "doc_hit@k": round(sum(doc_hit) / len(questions), 3),
                "evidence_hit@k": round(sum(evidence_hit) / len(questions), 3),
                "mrr": round(sum(rr) / len(questions), 3),
                "poisoned@k": round(sum(poisoned) / len(questions), 3),
                "context_chars": int(sum(chars) / len(questions)),
            })
    table = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "retrieval_chunk_sizes.csv", index=False)
    (RESULTS / "retrieval_chunk_sizes.md").write_text(
        f"# Retrieval by chunk size ({len(questions)} answerable questions)\n\n"
        + table.to_markdown(index=False) + "\n")
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    main()
