"""Turn the experiment results into charts (results/charts/*.png).

Reads whatever exists:
  results/retrieval_chunk_sizes.csv  (scripts/eval_retrieval.py)
  results/raw/*.jsonl                (scripts/run_experiments.py)

One metric per chart (no double axes), single colour, values written on bars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_assistant.evaluation import load_questions, summarize  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
CHARTS = RESULTS / "charts"
BAR = "#2a78d6"          # one series -> one colour
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def style(ax, title: str, ylabel: str, percent: bool) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if percent:
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def bar_chart(labels, values, title, ylabel, path, percent=True) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=SURFACE)
    bars = ax.bar(labels, values, color=BAR, width=0.55)
    for bar, value in zip(bars, values):
        text = f"{value:.0%}" if percent else f"{value:.2f}"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    style(ax, title, ylabel, percent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path.relative_to(RESULTS.parent))


def retrieval_charts() -> None:
    path = RESULTS / "retrieval_chunk_sizes.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    # Compare chunk sizes at about the same context budget (~3-4k characters):
    # many small chunks vs a few big ones.
    budget = [(500, 8), (1000, 5), (2000, 3)]
    rows = pd.concat([table[(table.chunk_size == s) & (table.k == k)] for s, k in budget])
    labels = [f"{s} chars × top {k}" for s, k in budget]
    bar_chart(labels, rows["evidence_hit@k"], "Answer passage retrieved, similar context budget",
              "evidence hit", CHARTS / "retrieval_evidence_hit.png")
    bar_chart(labels, rows["context_chars"] / 1000, "Context sent to the LLM per search",
              "thousand characters", CHARTS / "retrieval_context_size.png", percent=False)


COLUMNS = ["questions", "answer_accuracy", "correct_refusal_rate", "false_refusal_rate",
           "hallucination_rate", "citation_hit_rate", "citation_precision", "retrieval_recall",
           "avg_retries", "avg_llm_calls", "latency_mean_s", "latency_p95_s", "avg_tokens",
           "cost_per_question_usd"]


def experiment_tables() -> pd.DataFrame | None:
    """Rebuild the summary from the per-question files (the source of truth).

    An experiment that stopped early (e.g. daily API quota) is still listed, and
    additionally compared with the baseline on exactly the same questions.
    """
    files = sorted((RESULTS / "raw").glob("*.jsonl"))
    if not files:
        return None
    records = {f.stem: [json.loads(line) for line in f.read_text().splitlines()] for f in files}
    total = len(load_questions())
    rows = {name: summarize(recs) for name, recs in records.items()}
    table = pd.DataFrame(rows).T[COLUMNS].rename_axis("experiment")
    table.to_csv(RESULTS / "summary.csv")
    text = f"# End-to-end experiments ({total} questions)\n\n" + table.to_markdown() + "\n"

    baseline = {r["id"]: r for r in records.get("baseline", [])}
    for name, recs in records.items():
        if name != "baseline" and len(recs) < total and baseline:
            same = [baseline[r["id"]] for r in recs if r["id"] in baseline]
            paired = pd.DataFrame({name: summarize(recs), "baseline (same questions)": summarize(same)}).T
            text += (f"\n## {name} (stopped after {len(recs)} of {total} questions)\n\n"
                     "Compared with the baseline on exactly the same questions:\n\n"
                     + paired[COLUMNS].to_markdown() + "\n")
    (RESULTS / "summary.md").write_text(text)
    print("wrote results/summary.md")
    return table[table["questions"] == total]   # charts: complete experiments only


def experiment_charts(table: pd.DataFrame) -> None:
    names = list(table.index)
    for metric, title, percent, scale in [
        ("answer_accuracy", "Answer accuracy (LLM judge, answerable questions)", True, 1),
        ("avg_llm_calls", "LLM calls per question", False, 1),
        ("cost_per_question_usd", "Cost per 1,000 questions (US$)", False, 1000),
        ("latency_mean_s", "Mean latency per question (s, incl. rate-limit waits)", False, 1),
    ]:
        bar_chart(names, table[metric].astype(float).fillna(0) * scale, title, metric,
                  CHARTS / f"exp_{metric}.png", percent)


if __name__ == "__main__":
    CHARTS.mkdir(parents=True, exist_ok=True)
    retrieval_charts()
    complete = experiment_tables()
    if complete is not None and not complete.empty:
        experiment_charts(complete)
