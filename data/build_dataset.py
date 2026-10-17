"""Assemble the SFT dataset from all sources, then dedupe, decontaminate and split it.

Sources (priority order for de-duplication):
  generator   verified OS / CN / DBMS problems   data/processed/gen_{os,cn,dbms}_train.jsonl
  gate        GATE CSE PYQs (<=2020 + half of 2021-26) with explanations that reach the official key
              (data/build_gate.py; git-ignored, downloaded at build time)
  teacher     concept Q&A from build_concepts.py data/processed/concepts.jsonl (optional)
  textbook    Q&A grounded in open-licensed textbooks (data/build_textbook_qa.py; optional)
  mbpp        MBPP train split; kept only if the reference code passes its own tests
  spider      Spider train; the gold query must execute on its SQLite database (<= 15 per database)
  codealpaca  CodeAlpaca-20k; Python/DSA items whose code parses
  replay      alpaca-cleaned general instructions (limits forgetting)

Cleaning:
  * exact de-duplication of user prompts across all sources
  * near-duplicate removal (MinHash LSH, Jaccard >= 0.8) inside the public sources
  * decontamination: drop items sharing a 13-word n-gram with any evaluation item
    (CS-Bench English test, MBPP test, held-out generator problems, Spider dev, GATE test half),
    or containing a short evaluation item verbatim
  * length filter: <= 1024 tokens with the Qwen chat template

Usage:
  python data/build_dataset.py            # writes data/external/sft/sft_{train,val}_full.jsonl (training input),
                                          # public copies without GATE in data/processed/, + build_report.json
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generators.common import SYSTEM_PROMPT, write_jsonl  # noqa: E402
from eval.run_eval import (CODE_INSTR, CSB_URL, GATE_TEST, SPIDER_DIR, _download, _exec_sql,  # noqa: E402
                           _read_jsonl, spider_prompt)

PROC = ROOT / "data" / "processed"
PRIORITY = ["generator", "gate", "teacher", "textbook", "mbpp", "spider", "codealpaca", "replay"]
GATE_TRAIN = ROOT / "data" / "external" / "gate" / "gate_train.jsonl"
TEXTBOOK_QA = PROC / "textbook_qa.jsonl"
DSA_WORDS = re.compile(r"\b(array|list|linked list|stack|queue|tree|graph|sort|search|recurs|matrix|string|"
                       r"hash|dictionary|binary|prime|fibonacci|factorial|palindrome|algorithm|substring|"
                       r"duplicate|maximum|minimum|sum|reverse)", re.I)


def record(source: str, subject: str, topic: str, user: str, assistant: str, rid: str) -> dict:
    return {"id": rid, "subject": subject, "topic": topic, "source": source,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def load_generators() -> list[dict]:
    out = []
    for sub in ("os", "cn", "dbms"):
        for r in _read_jsonl(PROC / f"gen_{sub}_train.jsonl"):
            r = {k: v for k, v in r.items() if k != "final"}
            r["source"] = "generator"
            out.append(r)
    return out


def load_teacher() -> list[dict]:
    path = PROC / "concepts.jsonl"
    if not path.exists():
        print("  (no concepts.jsonl yet: run data/build_concepts.py; continuing without teacher data)")
        return []
    out = []
    for r in _read_jsonl(path):
        r["source"] = "teacher"
        out.append(r)
    return out


def _passes(code: str, timeout: int = 10) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def load_mbpp() -> list[dict]:
    from datasets import load_dataset
    out = []
    for r in load_dataset("google-research-datasets/mbpp", "full", split="train"):
        code = r["code"].replace("\r\n", "\n").replace("\t", "    ")
        if not _passes("\n".join([r["test_setup_code"] or "", code, *r["test_list"]])):
            continue
        user = f"{r['text']}\nYour code should pass these tests:\n" + "\n".join(r["test_list"]) + f"\n\n{CODE_INSTR}"
        out.append(record("mbpp", "Programming", "code/mbpp", user, f"```python\n{code.strip()}\n```",
                          f"mbpp-{r['task_id']}"))
    return out


def load_spider(n: int, rng: random.Random, per_db: int = 15) -> list[dict]:
    """Spider train questions; the gold query must execute on its SQLite database (at most `per_db` per database)."""
    rows = json.loads((SPIDER_DIR / "train_spider.json").read_text(encoding="utf-8"))
    rng.shuffle(rows)
    out, per = [], Counter()
    for i, r in enumerate(rows):
        if per[r["db_id"]] >= per_db:
            continue
        db = SPIDER_DIR / "database" / r["db_id"] / f"{r['db_id']}.sqlite"
        try:
            _exec_sql(db, r["query"])
        except Exception:
            continue
        per[r["db_id"]] += 1
        rec = record("spider", "DBMS", "sql/spider", spider_prompt(db, r["question"]),
                     f"```sql\n{r['query'].strip()}\n```", f"spider-{i}")
        # Prompts on one database share the whole schema, so duplicates are judged on the question alone.
        rec["dedupe_text"] = f"{r['db_id']}: {r['question']}"
        out.append(rec)
        if len(out) >= n:
            break
    return out


def load_gate() -> list[dict]:
    if not GATE_TRAIN.exists():
        print("  (no GATE explanations yet: run data/build_gate.py --explain; continuing without GATE)")
        return []
    return _read_jsonl(GATE_TRAIN)


def load_textbook() -> list[dict]:
    if not TEXTBOOK_QA.exists():
        print("  (no textbook Q&A yet: run data/build_textbook_qa.py; continuing without it)")
        return []
    return _read_jsonl(TEXTBOOK_QA)


def load_codealpaca(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    rows = list(load_dataset("sahil2801/CodeAlpaca-20k", split="train"))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        code = r["output"].strip()
        if "def " not in code or not DSA_WORDS.search(r["instruction"]):
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        user = r["instruction"].strip() + (f"\n\nInput: {r['input'].strip()}" if r["input"].strip() else "")
        out.append(record("codealpaca", "Programming", "code/codealpaca", user, f"```python\n{code}\n```",
                          f"codealpaca-{i}"))
        if len(out) >= n:
            break
    return out


def load_replay(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    rows = [r for r in load_dataset("yahma/alpaca-cleaned", split="train") if 80 <= len(r["output"]) <= 1500]
    out = []
    for i, r in enumerate(rng.sample(rows, n)):
        user = r["instruction"].strip() + (f"\n\n{r['input'].strip()}" if r["input"].strip() else "")
        out.append(record("replay", "General", "general/alpaca", user, r["output"].strip(), f"replay-{i}"))
    return out


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(ws: list[str], n: int) -> set[tuple]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


class MinHashLSH:
    """MinHash with banded LSH over word 5-gram shingles; verifies candidates with exact Jaccard."""

    P = (1 << 61) - 1

    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 0):
        rng = random.Random(seed)
        self.perms = [(rng.randrange(1, self.P), rng.randrange(0, self.P)) for _ in range(num_perm)]
        self.bands, self.rows = bands, num_perm // bands
        self.buckets: list[dict] = [defaultdict(list) for _ in range(bands)]
        self.shingles: list[set] = []

    @staticmethod
    def shingle(text: str) -> set:
        ws = words(text)
        return {zlib.crc32(" ".join(g).encode()) for g in ngrams(ws, 5)} or {zlib.crc32(" ".join(ws).encode())}

    def is_near_duplicate(self, text: str, threshold: float = 0.8) -> bool:
        sh = self.shingle(text)
        sig = [min((a * x + b) % self.P for x in sh) for a, b in self.perms]
        keys = [tuple(sig[i * self.rows:(i + 1) * self.rows]) for i in range(self.bands)]
        cands = {j for band, key in enumerate(keys) for j in self.buckets[band].get(key, [])}
        if any(len(sh & self.shingles[j]) / len(sh | self.shingles[j]) >= threshold for j in cands):
            return True
        idx = len(self.shingles)
        self.shingles.append(sh)
        for band, key in enumerate(keys):
            self.buckets[band][key].append(idx)
        return False


def eval_texts() -> dict[str, list[str]]:
    """Texts of every evaluation item, grouped by benchmark."""
    from datasets import load_dataset
    csb = _read_jsonl(_download(CSB_URL.format(split="test"), ROOT / "eval" / ".cache" / "csbench_test.jsonl"))
    texts = {"csbench": [" ".join(str(r.get(k, "")) for k in ("Question", "A", "B", "C", "D")) for r in csb],
             "mbpp_test": [r["text"] + " " + r["code"] + " " + " ".join(r["test_list"])
                           for r in load_dataset("google-research-datasets/mbpp", "full", split="test")]}
    for sub in ("os", "cn", "dbms"):
        texts[f"gen_{sub}_test"] = [r["messages"][1]["content"] for r in _read_jsonl(PROC / f"gen_{sub}_test.jsonl")]
    if (SPIDER_DIR / "dev.json").exists():
        texts["spider_dev"] = [r["question"] for r in json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))]
    if GATE_TEST.exists():
        texts["gate_test"] = [" ".join(r[k] for k in ("question", "A", "B", "C", "D")) for r in _read_jsonl(GATE_TEST)]
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spider-n", type=int, default=1500)
    ap.add_argument("--codealpaca-n", type=int, default=1200)
    ap.add_argument("--replay-n", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("loading sources ...", flush=True)
    raw = {"generator": load_generators(), "gate": load_gate(), "teacher": load_teacher(),
           "textbook": load_textbook(), "mbpp": load_mbpp(), "spider": load_spider(args.spider_n, rng),
           "codealpaca": load_codealpaca(args.codealpaca_n, rng), "replay": load_replay(args.replay_n, rng)}
    report = {"loaded": {k: len(v) for k, v in raw.items()}, "dropped": Counter()}
    print("  loaded:", report["loaded"], flush=True)

    print("building evaluation n-gram index ...", flush=True)
    ev = eval_texts()
    ev13 = set().union(*(ngrams(words(t), 13) for ts in ev.values() for t in ts))
    ev13_no_gen = set().union(*(ngrams(words(t), 13) for k, ts in ev.items() if not k.startswith("gen_") for t in ts))
    short_eval = {norm(t) for k in ("csbench", "spider_dev", "gate_test") for t in ev.get(k, [])
                  if 6 <= len(words(t)) < 13}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    seen, lsh, kept = set(), MinHashLSH(), []
    for source in PRIORITY:
        for r in raw[source]:
            user, full = r["messages"][1]["content"], r["messages"][1]["content"] + " " + r["messages"][2]["content"]
            dedupe_text = r.pop("dedupe_text", user)
            key = norm(dedupe_text)
            if key in seen:
                report["dropped"][f"{source}:exact_duplicate"] += 1
                continue
            if source != "generator" and lsh.is_near_duplicate(dedupe_text):
                report["dropped"][f"{source}:near_duplicate"] += 1
                continue
            # Generator problems share template sentences with the held-out generator tests by design
            # (their exact questions never overlap), so they are only checked against the other benchmarks.
            index = ev13_no_gen if source == "generator" else ev13
            if ngrams(words(full), 13) & index or any(s in norm(full) for s in short_eval):
                report["dropped"][f"{source}:contamination"] += 1
                continue
            text = tok.apply_chat_template(r["messages"], tokenize=False)
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            if n_tok > args.max_tokens:
                report["dropped"][f"{source}:too_long"] += 1
                continue
            seen.add(key)
            r["n_tokens"] = n_tok
            kept.append(r)

    by_source = defaultdict(list)
    for r in kept:
        by_source[r["source"]].append(r)
    train, val = [], []
    for source, rs in by_source.items():
        rng.shuffle(rs)
        k = max(1, round(len(rs) * args.val_frac))
        val += rs[:k]
        train += rs[k:]
    rng.shuffle(train)
    rng.shuffle(val)
    # Full splits (used for training) live in the git-ignored data/external/. The committed copies in
    # data/processed/ leave out GATE items, whose question text must not be republished.
    full_dir = ROOT / "data" / "external" / "sft"
    write_jsonl(train, full_dir / "sft_train_full.jsonl")
    write_jsonl(val, full_dir / "sft_val_full.jsonl")
    write_jsonl([r for r in train if r["source"] != "gate"], PROC / "sft_train.jsonl")
    write_jsonl([r for r in val if r["source"] != "gate"], PROC / "sft_val.jsonl")

    toks = [r["n_tokens"] for r in kept]
    report.update({
        "dropped": dict(sorted(report["dropped"].items())),
        "kept": {s: len(rs) for s, rs in sorted(by_source.items())},
        "kept_by_subject": dict(sorted(Counter(r["subject"] for r in kept).items())),
        "train": len(train), "val": len(val),
        "tokens": {"total": sum(toks), "mean": round(sum(toks) / len(toks)), "max": max(toks),
                   "train_total": sum(r["n_tokens"] for r in train)},
    })
    (PROC / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
