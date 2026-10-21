"""Tally hand-labelled errors into a Markdown table.

  python eval/error_report.py --run FT-B

Reads eval/error_analysis/labels_<run>.jsonl (id, category, note) and prints counts per task.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "procedure": "Procedure / state-tracking error: knows the method, executes it wrongly",
    "arithmetic": "Arithmetic or unit slip: correct method and setup, wrong computation",
    "knowledge": "Knowledge gap: wrong fact or definition",
    "schema_hallucination": "Schema hallucination: invents or misplaces tables, columns or conditions",
    "misread": "Misread task: solves a different problem than asked",
    "inconsistent": "Inconsistent: the final answer contradicts the model's own reasoning",
    "ambiguous": "Ambiguous item: the reference answer is debatable",
}
TASKS = ["csbench", "gen_os", "gen_cn", "gen_dbms", "mbpp", "spider"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="FT-B")
    args = ap.parse_args()
    path = ROOT / "eval" / "error_analysis" / f"labels_{args.run.lower()}.jsonl"
    with open(path, encoding="utf-8") as f:
        labels = [json.loads(l) for l in f if l.strip()]
    unknown = {l["category"] for l in labels} - set(CATEGORIES)
    assert not unknown, f"unknown categories: {unknown}"
    counts = Counter((l["id"].rsplit("-", 1)[0], l["category"]) for l in labels)
    total = Counter(l["category"] for l in labels)
    print("| Category | " + " | ".join(TASKS) + " | Total |")
    print("|---|" + "---|" * (len(TASKS) + 1))
    for cat in sorted(CATEGORIES, key=lambda c: -total[c]):
        cells = [str(counts[(t, cat)] or "·") for t in TASKS]
        print(f"| {cat} | " + " | ".join(cells) + f" | **{total[cat]}** ({100 * total[cat] / len(labels):.0f}%) |")
    print(f"| **n** | " + " | ".join(str(sum(counts[(t, c)] for c in CATEGORIES)) for t in TASKS)
          + f" | **{len(labels)}** |")


if __name__ == "__main__":
    main()
