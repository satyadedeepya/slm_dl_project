"""Build a Markdown comparison of evaluation runs.

Reads eval/results/<run>/summary.json for every run (smoke tests excluded) and
writes eval/results/RESULTS.md with an overall table and per-group breakdowns.

Usage:
  python eval/summarize.py                     # all runs
  python eval/summarize.py B0_zeroshot B1_3shot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TASK_NAMES = {
    "csbench": "CS-Bench (DSA/OS/CN, MCQ + T/F)",
    "gen_os": "Verified OS numericals (exact match)",
    "mbpp": "MBPP (pass@1)",
    "spider": "Spider dev (execution acc.)",
    "gate": "GATE MCQ",
}


def load(runs: list[str]) -> dict[str, dict]:
    if not runs:
        runs = sorted(p.parent.name for p in RESULTS.glob("*/summary.json") if not p.parent.name.startswith("smoke"))
    return {r: json.loads((RESULTS / r / "summary.json").read_text(encoding="utf-8")) for r in runs}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def render(data: dict[str, dict]) -> str:
    runs = list(data)
    tasks = [t for t in TASK_NAMES if any(t in d["results"] for d in data.values())]
    out = ["# Evaluation results", "",
           "Accuracy in %, with the 95% bootstrap confidence interval in brackets.", ""]

    out += ["| Task | n | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
    for t in tasks:
        n = next(d["results"][t]["n"] for d in data.values() if t in d["results"])
        cells = []
        for r in runs:
            res = data[r]["results"].get(t)
            cells.append(f"{pct(res['acc'])} [{pct(res['ci95'][0])}–{pct(res['ci95'][1])}]" if res else "–")
        out.append(f"| {TASK_NAMES[t]} | {n} | " + " | ".join(cells) + " |")

    for t in tasks:
        groups = sorted({g for d in data.values() for g in d["results"].get(t, {}).get("by_group", {})})
        if not groups:
            continue
        out += ["", f"## {TASK_NAMES[t]}: breakdown", "",
                "| Group | n | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
        for g in groups:
            n = next(d["results"][t]["by_group"][g]["n"] for d in data.values()
                     if g in d["results"].get(t, {}).get("by_group", {}))
            cells = [pct(data[r]["results"][t]["by_group"][g]["acc"])
                     if g in data[r]["results"].get(t, {}).get("by_group", {}) else "–" for r in runs]
            out.append(f"| {g} | {n} | " + " | ".join(cells) + " |")

    out += ["", "## Run settings", "", "| Run | Model | Adapter | Few-shot |", "|---|---|---|---|"]
    for r in runs:
        m = data[r]["meta"]
        out.append(f"| {r} | {m['model']} | {m['adapter'] or '–'} | {m['fewshot']} |")
    return "\n".join(out) + "\n"


def main():
    data = load(sys.argv[1:])
    if not data:
        raise SystemExit("no runs with summary.json found in eval/results/")
    md = render(data)
    (RESULTS / "RESULTS.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
