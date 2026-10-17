"""Assemble the SFT dataset from all sources, then dedupe, decontaminate and split it.

Sources (priority order for de-duplication):
  generator   verified OS / CN / DBMS problems   data/processed/gen_{os,cn,dbms}_train.jsonl
  teacher     concept Q&A from build_concepts.py data/processed/concepts.jsonl (optional)
  mbpp        MBPP train split; kept only if the reference code passes its own tests
  sql         b-mc2/sql-create-context; complex queries only; kept only if SQLite accepts the query
  codealpaca  CodeAlpaca-20k; Python/DSA items whose code parses
  replay      alpaca-cleaned general instructions (limits forgetting)

Cleaning:
  * exact de-duplication of user prompts across all sources
  * near-duplicate removal (MinHash LSH, Jaccard >= 0.8) inside the public sources
  * decontamination: drop items sharing a 13-word n-gram with any evaluation item
    (CS-Bench English test, MBPP test, held-out generator problems), or containing a
    short evaluation item verbatim
  * length filter: <= 1024 tokens with the Qwen chat template

Usage:
  python data/build_dataset.py            # writes data/processed/sft_{train,val}.jsonl + build_report.json
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generators.common import SYSTEM_PROMPT, write_jsonl  # noqa: E402
from eval.run_eval import CODE_INSTR, CSB_URL, SQL_INSTR, _download, _read_jsonl  # noqa: E402

PROC = ROOT / "data" / "processed"
PRIORITY = ["generator", "teacher", "mbpp", "sql", "codealpaca", "replay"]
SQL_COMPLEX = re.compile(r"\b(join|group by|order by|having|intersect|union|except)\b|\(\s*select", re.I)
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


def _sql_ok(ddl: str, query: str) -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ddl)
        conn.execute(f"EXPLAIN {query}")
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def load_sql(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    rows = [r for r in load_dataset("b-mc2/sql-create-context", split="train") if SQL_COMPLEX.search(r["answer"])]
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        ddl = r["context"].replace("; ", ";\n").strip()
        if not ddl.endswith(";"):
            ddl += ";"
        if not _sql_ok(ddl, r["answer"]):
            continue
        user = f"Database schema:\n{ddl}\n\nQuestion: {r['question']}\n\n{SQL_INSTR}"
        out.append(record("sql", "DBMS", "sql/text_to_sql", user, f"```sql\n{r['answer'].strip()}\n```", f"sql-{i}"))
        if len(out) >= n:
            break
    return out


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
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sql-n", type=int, default=1000)
    ap.add_argument("--codealpaca-n", type=int, default=1200)
    ap.add_argument("--replay-n", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("loading sources ...", flush=True)
    raw = {"generator": load_generators(), "teacher": load_teacher(), "mbpp": load_mbpp(),
           "sql": load_sql(args.sql_n, rng), "codealpaca": load_codealpaca(args.codealpaca_n, rng),
           "replay": load_replay(args.replay_n, rng)}
    report = {"loaded": {k: len(v) for k, v in raw.items()}, "dropped": Counter()}
    print("  loaded:", report["loaded"], flush=True)

    print("building evaluation n-gram index ...", flush=True)
    ev = eval_texts()
    ev13 = set().union(*(ngrams(words(t), 13) for ts in ev.values() for t in ts))
    ev13_no_gen = set().union(*(ngrams(words(t), 13) for k, ts in ev.items() if not k.startswith("gen_") for t in ts))
    short_eval = {norm(t) for t in ev["csbench"] if 6 <= len(words(t)) < 13}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    seen, lsh, kept = set(), MinHashLSH(), []
    for source in PRIORITY:
        for r in raw[source]:
            user, full = r["messages"][1]["content"], r["messages"][1]["content"] + " " + r["messages"][2]["content"]
            key = norm(user)
            if key in seen:
                report["dropped"][f"{source}:exact_duplicate"] += 1
                continue
            if source != "generator" and lsh.is_near_duplicate(user):
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
    write_jsonl(train, PROC / "sft_train.jsonl")
    write_jsonl(val, PROC / "sft_val.jsonl")

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
