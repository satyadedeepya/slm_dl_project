"""Draw a stratified sample of wrong answers from one run for manual error analysis.

  python eval/error_sample.py --run FT-B --out eval/error_analysis/sample_ft-b.jsonl

Each record has the task, question, reference answer, the model's output and diagnostic hints
(SQL or Python error). GATE is excluded: its exam text is not redistributed.
Labels are added by hand in eval/error_analysis/labels_<run>.jsonl and tallied by
eval/error_report.py.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.run_eval import SPIDER_DIR, extract_block  # noqa: E402

QUOTA = {"csbench": 12, "gen_os": 8, "gen_cn": 8, "gen_dbms": 8, "mbpp": 7, "spider": 7}


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def references(task: str) -> list[dict] | None:
    """Reference answers aligned with the result rows (same item order as eval/run_eval.py)."""
    if task.startswith("gen_"):
        rows = read(ROOT / "data" / "processed" / f"gen_{task[4:]}_test.jsonl")
        return [{"reference": r["messages"][2]["content"].split("**Final answer:**")[-1].strip(),
                 "topic": r["topic"]} for r in rows]
    if task == "spider":
        dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
        return [{"reference": r["query"], "db_id": r["db_id"]} for r in random.Random(0).sample(dev, 200)]
    if task == "mbpp":
        from datasets import load_dataset
        test = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
        return [{"reference": "\n".join(r["test_list"]), "setup": r.get("test_setup_code") or ""}
                for r in random.Random(0).sample(test, 200)]
    return None


def python_error(code: str, tests: str, setup: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("\n".join([setup, code, tests]))
        path = f.name
    try:
        res = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        return (res.stderr.strip().splitlines() or ["(no error)"])[-1][:200]
    except subprocess.TimeoutExpired:
        return "timeout"
    finally:
        Path(path).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="FT-B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out or ROOT / "eval" / "error_analysis" / f"sample_{args.run.lower()}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sample = []
    for task, k in QUOTA.items():
        rows = read(ROOT / "eval" / "results" / args.run / f"{task}.jsonl")
        refs = references(task)
        wrong = [i for i, r in enumerate(rows) if not r["correct"]]
        for i in sorted(rng.sample(wrong, min(k, len(wrong)))):
            r = rows[i]
            rec = {"id": f"{task}-{i}", "task": task, "prompt": r["prompt"], "output": r["output"]}
            if task == "csbench":
                rec.update(reference=r["gold"], pred=r["pred"])
            elif refs:
                rec.update(refs[i])
            if task == "spider":
                rec["pred_sql"] = r.get("sql")
            if task == "mbpp":
                rec["error"] = python_error(extract_block(r["output"], "python"), rec["reference"], rec.pop("setup"))
            sample.append(rec)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(sample)} wrong answers to {out}")


if __name__ == "__main__":
    main()
