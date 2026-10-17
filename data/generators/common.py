"""Shared helpers for the verified problem generators.

Every generator returns a `Problem`: a natural-language question, a step-by-step
solution whose numbers come from actually running the algorithm, and a
machine-readable `final` dict that `grading.grade` uses to check model output.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

SYSTEM_PROMPT = (
    "You are CSE-Tutor, an expert teaching assistant for Computer Science "
    "(DSA, Programming, DBMS, Operating Systems, Computer Networks). "
    "Explain step by step and end numerical answers with a line starting "
    "with 'Final answer:'."
)


@dataclass
class Problem:
    subject: str          # "OS", "CN", "DBMS"
    topic: str            # e.g. "cpu_scheduling/srtf"
    question: str
    solution: str         # step-by-step answer ending in "**Final answer:** ..."
    final: dict = field(default_factory=dict)  # grading spec, see grading.py

    def to_record(self, uid: str) -> dict:
        return {
            "id": uid,
            "subject": self.subject,
            "topic": self.topic,
            "source": "generator",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.solution},
            ],
            "final": self.final,
        }


def fmt(x) -> str:
    """Format an int/Fraction/float: integers stay integers, others get 2 decimals."""
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        return f"{float(x):.2f}"
    if isinstance(x, float) and not x.is_integer():
        return f"{x:.2f}"
    return str(int(x))


def fmt_avg(total: int, n: int) -> str:
    """'(sum)/n = value' with the exact fraction kept visible for teaching."""
    v = Fraction(total, n)
    return f"{total}/{n} = {fmt(v)}" if v.denominator != 1 else f"{total}/{n} = {v.numerator}"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")


def parse_numbers(text: str) -> list[float]:
    """All numbers in `text`, with 'a/b' fractions evaluated."""
    out = []
    for m in _NUM.finditer(text):
        tok = m.group(0).replace(" ", "")
        if "/" in tok:
            a, b = tok.split("/")
            out.append(float(a) / float(b) if float(b) else float("nan"))
        else:
            out.append(float(tok))
    return out


def final_segment(text: str) -> str:
    """Text after the last 'Final answer' marker (or the tail if none)."""
    idx = text.lower().rfind("final answer")
    return text[idx:] if idx != -1 else text[-300:]


def run_generators(generators, prefix: str, n: int, seed: int, split: str = "train") -> list[dict]:
    """Draw `n` unique problems from weighted (generator, weight) pairs, deterministically for a seed."""
    rng = random.Random(seed)
    funcs, weights = zip(*generators)
    records, seen = [], set()
    while len(records) < n:
        prob = rng.choices(funcs, weights)[0](rng, split)
        if prob.question in seen:
            continue
        seen.add(prob.question)
        records.append(prob.to_record(f"{prefix}-{split}-{seed}-{len(records):05d}"))
    return records


def cli(generators, prefix: str, label: str, default_n: int) -> None:
    ap = argparse.ArgumentParser(description=f"Generate verified {label} problems.")
    ap.add_argument("--n", type=int, default=default_n)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = run_generators(generators, prefix, args.n, args.seed, args.split)
    write_jsonl(recs, args.out)
    print(f"wrote {len(recs)} {label} problems to {args.out}")
