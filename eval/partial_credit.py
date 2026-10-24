"""Value-level partial credit for the verified numerical tasks.

Exact match needs every number in the final answer to be right. This script also reports the share of
expected values that appear in the model's final answer (same tolerance as the grader), which separates
"knows the method, slips on one number" from "wrong method".

  python eval/partial_credit.py --runs B0_zeroshot FT-B FT-C --tasks gen_dl gen_dbms gen_cn
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generators.common import final_segment, parse_numbers  # noqa: E402
from data.generators.grading import TOL  # noqa: E402


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def value_credit(spec: dict, output: str) -> float | None:
    if spec.get("type") != "numbers":
        return None
    found = parse_numbers(final_segment(output))
    hits = 0
    for v in spec["values"]:
        tol = spec["rel_tol"] * abs(v) if spec.get("rel_tol") else TOL
        hits += any(abs(x - v) <= max(tol, 1e-9) for x in found)
    return hits / len(spec["values"])


def bootstrap(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def paired_permutation(a: list[float], b: list[float], n: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on per-item differences."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    obs = abs(sum(d))
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["B0_zeroshot", "FT-B", "FT-C"])
    ap.add_argument("--tasks", nargs="+", default=["gen_dl", "gen_cn"])
    args = ap.parse_args()
    print("| Task | " + " | ".join(args.runs) + " |")
    print("|---|" + "---|" * len(args.runs))
    for task in args.tasks:
        items = read(ROOT / "data" / "processed" / f"gen_{task[4:]}_test.jsonl")
        cells, per_run = [], {}
        for run in args.runs:
            path = ROOT / "eval" / "results" / run / f"{task}.jsonl"
            if not path.exists():
                cells.append("–")
                continue
            credits = [c for it, r in zip(items, read(path)) if (c := value_credit(it["final"], r["output"])) is not None]
            if not credits:  # e.g. gen_dbms answers are attribute sets, not numbers
                cells.append("n/a")
                continue
            per_run[run] = credits
            lo, hi = bootstrap(credits)
            cells.append(f"{100 * sum(credits) / len(credits):.1f} [{100 * lo:.1f}–{100 * hi:.1f}]")
        print(f"| {task} (n with numeric answers = {len(credits)}) | " + " | ".join(cells) + " |")
        runs = list(per_run)
        for prev, last in zip(runs, runs[1:]):
            print(f"    paired permutation p ({prev} vs {last}) = {paired_permutation(per_run[prev], per_run[last]):.4f}")
        if len(runs) > 2:
            print(f"    paired permutation p ({runs[0]} vs {runs[-1]}) = "
                  f"{paired_permutation(per_run[runs[0]], per_run[runs[-1]]):.4f}")


if __name__ == "__main__":
    main()
