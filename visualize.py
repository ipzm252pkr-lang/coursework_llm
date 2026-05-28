from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "BufferMemory":  "#e74c3c",
    "SummaryMemory": "#3498db",
    "VectorMemory":  "#2ecc71",
}
LABELS = {
    "BufferMemory":  "Buffer (Sliding Window)",
    "SummaryMemory": "Summary (LLM Compression)",
    "VectorMemory":  "Vector (Semantic Search)",
}
DPI = 150


def load(path: str = "results.json") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found - run experiment.py first")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig_retention(data: dict) -> None:
    retention = data["retention"]
    n_list = sorted(int(k) for k in retention)

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, color in COLORS.items():
        scores = [retention[str(n)].get(strategy, {}).get("retention_score", np.nan) for n in n_list]
        ax.plot(n_list, scores, marker="o", lw=2.2, ms=7, color=color, label=LABELS[strategy])

    ax.set(
        xlabel="Number of distractors",
        ylabel="Memory Retention Score",
        ylim=(-0.05, 1.05),
    )
    ax.set_xticks(n_list)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("fig1_retention.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved fig1_retention.png")


def fig_latency(data: dict) -> None:
    latency = data["latency"]
    strategies = [s for s in COLORS if s in latency and "error" not in latency[s]]
    if not strategies:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [latency[s]["latencies_ms"] for s in strategies],
        patch_artist=True,
        medianprops={"color": "black", "lw": 2},
    )
    for patch, s in zip(bp["boxes"], strategies):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.7)

    for i, s in enumerate(strategies, 1):
        mean = np.mean(latency[s]["latencies_ms"])
        ax.plot(i, mean, marker="D", color="white", markeredgecolor="black", ms=7, zorder=5)
        ax.annotate(f"μ={mean:.0f}", xy=(i, mean), xytext=(8, 4), textcoords="offset points", fontsize=9)

    ax.set_xticklabels([LABELS[s] for s in strategies], rotation=10, ha="right")
    ax.set(ylabel="Latency (ms)")
    ax.grid(axis="y", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("fig2_latency.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved fig2_latency.png")


def fig_context_growth(data: dict) -> None:
    ctx = data["context_growth"]
    strategies = [s for s in COLORS if s in ctx and "error" not in ctx[s]]
    if not strategies:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in strategies:
        ax.plot(ctx[s]["turns"], ctx[s]["context_sizes"], lw=2.2, color=COLORS[s], label=LABELS[s])

    ax.set(
        xlabel="Turn",
        ylabel="Context size (tokens, estimated)",
    )
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("fig3_context.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved fig3_context.png")


def fig_heatmap(data: dict, n_distractors: int = 15) -> None:
    retention = data["retention"]
    n_str = str(n_distractors) if str(n_distractors) in retention else sorted(retention)[-1]

    strategies = [s for s in COLORS if s in retention.get(n_str, {})]
    if not strategies:
        return

    q_labels = ["Name?", "Age?", "City?", "Job?", "Language?", "Pet name?", "Learning?"]
    matrix = []
    col_labels = []
    for s in strategies:
        scores = retention[n_str][s].get("per_question_scores", [])
        if scores:
            matrix.append(scores)
            col_labels.append(LABELS[s])

    if not matrix:
        return

    arr = np.array(matrix).T
    fig, ax = plt.subplots(figsize=(max(5, len(col_labels) * 2.5), max(4, arr.shape[0] * 0.8)))
    sns.heatmap(
        arr, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        xticklabels=col_labels, yticklabels=q_labels[:arr.shape[0]],
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Retention Score"},
    )
    ax.set(
        xlabel="Memory strategy",
        ylabel="Recall question",
    )
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig("fig4_heatmap.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved fig4_heatmap.png")


def fig_statistical(data: dict) -> None:
    stat = data["statistical"]
    if not stat or "error" in stat:
        return

    pairs = list(stat.keys())
    p_values = [stat[p]["p_value"] for p in pairs]
    significant = [stat[p]["significant"] for p in pairs]
    bar_colors = ["#2ecc71" if s else "#e74c3c" for s in significant]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        [p.replace("_vs_", " vs\n") for p in pairs],
        p_values, color=bar_colors, alpha=0.8, edgecolor="black",
    )
    ax.axvline(0.05, color="black", ls="--", lw=1.5, label="α = 0.05")

    for bar, p in zip(bars, p_values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"p={p:.4f}", va="center", fontsize=9)

    ax.set(
        xlabel="p-value",
        xlim=(0, max(p_values) * 1.4 + 0.1),
    )
    ax.legend()
    ax.grid(axis="x", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("fig5_stats.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved fig5_stats.png")


def main(results_path: str = "results.json") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    data = load(results_path)

    fig_retention(data)
    fig_latency(data)
    fig_context_growth(data)
    fig_heatmap(data, n_distractors=15)
    fig_statistical(data)

    saved = sorted(Path(".").glob("fig*.png"))
    print("Figures saved:")
    for f in saved:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
