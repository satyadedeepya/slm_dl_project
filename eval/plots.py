"""Figures for the reports, written to docs/figures/.

  python eval/plots.py

- accuracy.png: per-task accuracy with 95% CIs for B0, B1, FT-A, FT-B and FT-C (from eval/results/*/summary.json)
- loss.png: training and validation loss for FT-B and FT-A (from outputs/<run>/train_summary.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
FIGURES = ROOT / "docs" / "figures"

# Categorical slots in fixed order (validated: adjacent CVD ΔE >= 9.1 on the light surface).
# Aqua and yellow are below 3:1 contrast, so bars carry value labels and the reports carry tables.
RUNS = [("B0_zeroshot", "B0 base, zero-shot", "#2a78d6"),
        ("B1_3shot", "B1 base, 3-shot", "#eb6834"),
        ("FT-A", "FT-A (no generator data)", "#1baf7a"),
        ("FT-B", "FT-B (full mix)", "#eda100"),
        ("FT-C", "FT-C (FT-B + DL stage)", "#e87ba4")]
TASKS = [("csbench", "CS-Bench"), ("gen_os", "OS numericals"), ("gen_cn", "CN numericals"),
         ("gen_dbms", "DBMS FD problems"), ("gen_dl", "DL numericals"), ("mbpp", "MBPP"), ("spider", "Spider"), ("gate", "GATE 2021–26")]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy():
    runs = [r for r in RUNS if (RESULTS / r[0] / "summary.json").exists()]
    data = {r: json.loads((RESULTS / r / "summary.json").read_text(encoding="utf-8"))["results"] for r, _, _ in runs}
    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor=SURFACE)
    style(ax)
    width, gap = 0.8 / len(runs), 0.02
    for i, (run, label, color) in enumerate(runs):
        for j, (task, _) in enumerate(TASKS):
            res = data[run].get(task)
            if not res:
                continue
            x = j + (i - (len(runs) - 1) / 2) * width
            acc, (lo, hi) = 100 * res["acc"], (100 * res["ci95"][0], 100 * res["ci95"][1])
            ax.bar(x, acc, width - gap, color=color, label=label if j == 0 else None, zorder=2)
            ax.errorbar(x, acc, yerr=[[acc - lo], [hi - acc]], fmt="none", ecolor=INK_2, elinewidth=0.8,
                        capsize=0, zorder=3)
            ax.text(x, hi + 1.2, f"{acc:.0f}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(TASKS)), [t for _, t in TASKS], color=INK)
    ax.set_ylabel("Accuracy (%)", color=INK_2)
    ax.set_ylim(0, 80)
    ax.set_title("Accuracy per task, with 95% bootstrap CIs (missing bars: run not evaluated on that task)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, ncol=len(runs), loc="upper left", bbox_to_anchor=(0, 1.0), labelcolor=INK, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy.png", dpi=160)


def rolling(xs, k=6):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]


def loss():
    # One panel per run: the runs are validated on different samples (source exclusion, DL stage), so their
    # validation losses are not comparable and must not share an axis.
    panels = [(r, t) for r, t in (("ft-b", "FT-B (full mix)"), ("ft-a", "FT-A (no generator data)"),
                                  ("ft-c", "FT-C (DL stage, from FT-B)"))
              if (ROOT / "outputs" / r / "train_summary.json").exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2), facecolor=SURFACE)
    for ax, (run, title) in zip(axes, panels):
        style(ax)
        hist = json.loads((ROOT / "outputs" / run / "train_summary.json").read_text(encoding="utf-8"))["log_history"]
        train = sorted({h["step"]: h["loss"] for h in hist if "loss" in h}.items())
        val = sorted({h["step"]: h["eval_loss"] for h in hist if "eval_loss" in h}.items())
        steps, losses = zip(*train)
        ax.plot(steps, losses, color="#2a78d6", alpha=0.25, linewidth=1)
        ax.plot(steps, rolling(list(losses)), color="#2a78d6", linewidth=2, label="training loss (rolling mean of 6 logs)")
        vs, vl = zip(*val)
        ax.plot(vs, vl, color="#eb6834", linewidth=2, marker="o", markersize=5, label="validation loss")
        ax.annotate(f"{vl[-1]:.3f}", (vs[-1], vl[-1]), textcoords="offset points", xytext=(-4, 8),
                    ha="right", fontsize=8, color=INK_2)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Optimiser step (effective batch 32)", color=INK_2)
        ax.set_ylim(0, 1.5)
    axes[0].set_ylabel("Loss", color=INK_2)
    axes[0].legend(frameon=False, loc="lower left", labelcolor=INK, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "loss.png", dpi=160)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    accuracy()
    loss()
    print("wrote", *sorted(p.name for p in FIGURES.glob("*.png")))
