"""Turn the experiment results into charts (results/charts/*.png).

Reads whatever exists:
  results/retrieval_chunk_sizes.csv  (scripts/eval_retrieval.py)
  results/summary.csv                (scripts/run_experiments.py)

One metric per chart (no double axes), single colour, values written on bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

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


def experiment_charts() -> None:
    path = RESULTS / "summary.csv"
    if not path.exists():
        return
    table = pd.read_csv(path).set_index("experiment")
    names = list(table.index)
    for metric, title, percent in [
        ("answer_accuracy", "Answer accuracy (LLM judge, answerable questions)", True),
        ("citation_hit_rate", "Answers citing the correct document", True),
        ("hallucination_rate", "Hallucination rate (lower is better)", True),
        ("correct_refusal_rate", "Correct refusals on unanswerable questions", True),
        ("latency_mean_s", "Mean latency per question (seconds)", False),
        ("cost_per_question_usd", "Cost per question (US$)", False),
    ]:
        if metric in table:
            bar_chart(names, table[metric].fillna(0), title, metric, CHARTS / f"exp_{metric}.png", percent)


if __name__ == "__main__":
    CHARTS.mkdir(parents=True, exist_ok=True)
    retrieval_charts()
    experiment_charts()
