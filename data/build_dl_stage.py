"""Build the Deep Learning continued-training set for FT-C (FT-B + a DL stage).

  python data/build_dl_stage.py

Train mix:
  * verified DL generator problems (data/processed/gen_dl_train.jsonl)
  * DL concept Q&A from the Gemini teacher (subject "DL" in data/processed/concepts.jsonl), if present
  * replay: a random sample of the original SFT training mix, so the earlier skills are not forgotten
Validation: held-out DL generator problems plus a sample of the original validation split.

Writes data/external/sft/dl_stage_{train,val}.jsonl (git-ignored, as the replay part contains exam text)
and data/processed/dl_stage_report.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generators import dl_gen  # noqa: E402

PROC = ROOT / "data" / "processed"
SFT = ROOT / "data" / "external" / "sft"


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gen = read(PROC / "gen_dl_train.jsonl")
    test_q = {r["messages"][1]["content"] for r in read(PROC / "gen_dl_test.jsonl")}
    gen = [r for r in gen if r["messages"][1]["content"] not in test_q]
    concepts = [r for r in read(PROC / "concepts.jsonl") if r.get("subject") == "DL"]
    for r in gen:
        r.setdefault("source", "generator")
    replay = rng.sample(read(SFT / "sft_train_full.jsonl"), args.replay)
    train = gen + concepts + replay
    rng.shuffle(train)

    # Validation: fresh DL problems (different seed, train parameter pool) + original validation examples.
    seen = {r["messages"][1]["content"] for r in gen} | test_q
    dl_val = [r for r in dl_gen.generate(80, seed=777) if r["messages"][1]["content"] not in seen][:50]
    for r in dl_val:
        r["source"] = "generator"
    val = dl_val + rng.sample(read(SFT / "sft_val_full.jsonl"), 50)

    write(train, SFT / "dl_stage_train.jsonl")
    write(val, SFT / "dl_stage_val.jsonl")
    report = {"train": len(train), "val": len(val), "dl_generator": len(gen), "dl_concepts": len(concepts),
              "replay": len(replay), "train_by_source": Counter(r["source"] for r in train),
              "concept_formats": Counter(r.get("format") for r in concepts)}
    (PROC / "dl_stage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
