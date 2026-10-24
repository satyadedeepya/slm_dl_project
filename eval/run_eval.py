"""Evaluate a (base or fine-tuned) model on the held-out CSE benchmarks.

Tasks
  csbench  CS-Bench English test, DSA/OS/CN, multiple-choice + assertion (accuracy)
  gen_os   Held-out verified OS problems (data/processed/gen_os_test.jsonl, exact match)
  gen_cn   Held-out verified CN problems (gen_cn_test.jsonl)
  gen_dbms Held-out verified DBMS problems (gen_dbms_test.jsonl)
  mbpp     MBPP test split (pass@1 via executing the unit tests)
  spider   Spider dev (execution accuracy); Spider is downloaded to data/external/spider_data
  gate     Held-out GATE CSE MCQs (2021-2026 half) from data/build_gate.py

Examples (Colab T4):
  python eval/run_eval.py --run-name B0_zeroshot
  python eval/run_eval.py --run-name B1_3shot --fewshot 3
  python eval/run_eval.py --run-name FT-B --adapter outputs/ft-b/adapter
  python eval/run_eval.py --run-name smoke --limit 5 --tasks csbench gen_os
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generators import cn_gen, dbms_gen, dl_gen, os_gen  # noqa: E402
from data.generators.common import SYSTEM_PROMPT, write_jsonl  # noqa: E402
from data.generators.grading import grade  # noqa: E402

CSB_DOMAINS = {"Data Structure and Algorithm": "DSA", "Operating System": "OS", "Computer Network": "CN"}
CSB_URL = "https://huggingface.co/datasets/CS-Bench/CS-Bench/resolve/main/CS-Bench/English/{split}.jsonl"
CACHE = ROOT / "eval" / ".cache"

MCQ_INSTR = "Think briefly, then end your response with a line of the form 'Answer: X' where X is one of A, B, C, D."
TF_INSTR = "Decide whether the statement is true or false. Think briefly, then end with 'Answer: True' or 'Answer: False'."
GEN_INSTR = "Solve step by step. End with a line starting with 'Final answer:'."
SQL_INSTR = "Write one SQLite query that answers the question. Return only the SQL inside a ```sql``` block."
CODE_INSTR = "Write a Python function that solves the task. Return only the code inside a ```python``` block."


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Generator:
    def __init__(self, model: str, adapter: str | None, batch_size: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel Arc GPUs
            device = "xpu"
        else:
            device = "cpu"
        print(f"device: {device}", flush=True)
        # Load weights straight onto the device. On integrated GPUs (shared RAM) loading to CPU
        # first and then calling .to(device) briefly holds two full copies of the model.
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float32 if device == "cpu" else torch.float16, device_map={"": device})
        self.device = device
        if adapter:
            from peft import PeftModel
            # Merge LoRA weights into the base model so generation runs at base-model speed.
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self.batch_size = batch_size
        self.torch = torch

    def mem(self) -> str:
        """Peak accelerator memory so far, for spotting batch sizes that will not fit."""
        acc = getattr(self.torch, self.device, None) if self.device != "cpu" else None
        return f"  (peak {acc.max_memory_allocated() / 1e9:.1f} GB)" if acc else ""

    def chat(self, conversations: list[list[dict]], max_new_tokens: int) -> list[str]:
        prompts = [self.tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in conversations]
        # Sort by length so batches pad less; restore the order afterwards.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outs: dict[int, str] = {}
        for b in range(0, len(order), self.batch_size):
            ids = order[b:b + self.batch_size]
            enc = self.tok([prompts[i] for i in ids], return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for i, row in zip(ids, gen[:, enc["input_ids"].shape[1]:]):
                outs[i] = self.tok.decode(row, skip_special_tokens=True)
            print(f"  generated {min(b + self.batch_size, len(order))}/{len(order)}{self.mem()}", flush=True)
        return [outs[i] for i in range(len(prompts))]


def conv(user: str, shots: list[tuple[str, str]]) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in shots:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_choice(text: str) -> str | None:
    m = re.findall(r"answer\s*(?:is|:)?\s*[:\-]?\s*\**\s*\(?([ABCD])\b", text, re.I)
    if m:
        return m[-1].upper()
    m = re.match(r"\s*\(?([ABCD])[\).:\s]", text + " ")
    return m.group(1) if m else None


def extract_bool(text: str) -> bool | None:
    m = re.findall(r"answer\s*(?:is|:)?\s*[:\-]?\s*\**\s*(true|false)\b", text, re.I)
    if not m:
        m = re.findall(r"\b(true|false)\b", text[-200:], re.I)
    return None if not m else m[-1].lower() == "true"


def extract_block(text: str, lang: str) -> str:
    m = re.search(rf"```(?:{lang})?\s*\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    if lang == "sql":
        m = re.search(r"(select\b.*)", text, re.S | re.I)
        if m:
            return m.group(1).split(";")[0].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Tasks: each returns (items, conversations, max_new_tokens, scorer)
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _mcq_text(r: dict) -> str:
    return f"{r['Question']}\n\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n\n{MCQ_INSTR}"


def task_csbench(args):
    rows = [r for r in _read_jsonl(_download(CSB_URL.format(split="test"), CACHE / "csbench_test.jsonl"))
            if r["Domain"] in CSB_DOMAINS and r["Format"] in ("Multiple-choice", "Assertion")]
    valid = [r for r in _read_jsonl(_download(CSB_URL.format(split="valid"), CACHE / "csbench_valid.jsonl"))
             if r["Domain"] in CSB_DOMAINS and r["Format"] in ("Multiple-choice", "Assertion")]
    # Stratified sample: same fraction from every (domain, format) cell.
    rng = random.Random(0)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["Domain"], r["Format"])].append(r)
    frac = min(1.0, args.csbench_n / len(rows))
    items = []
    for key in sorted(cells):
        rng.shuffle(cells[key])
        items += cells[key][:max(1, round(len(cells[key]) * frac))]

    def text(r):
        return _mcq_text(r) if r["Format"] == "Multiple-choice" else f"Statement: {r['Question']}\n\n{TF_INSTR}"

    def gold(r):
        return str(r["Answer"]).strip() if r["Format"] == "Multiple-choice" else str(r["Answer"]).lower() == "true"

    def shot_answer(r):
        expl = (r.get("Explanation") or "").strip()
        ans = gold(r) if r["Format"] == "Multiple-choice" else ("True" if gold(r) else "False")
        return (expl + "\n\n" if expl else "") + f"Answer: {ans}"

    convs = []
    for r in items:
        pool = [v for v in valid if v["Domain"] == r["Domain"] and v["Format"] == r["Format"]]
        shots = [(text(v), shot_answer(v)) for v in random.Random(r["ID"]).sample(pool, min(args.fewshot, len(pool)))]
        convs.append(conv(text(r), shots))

    def score(r, out):
        pred = extract_choice(out) if r["Format"] == "Multiple-choice" else extract_bool(out)
        return pred == gold(r), {"pred": pred, "gold": gold(r),
                                 "group": [CSB_DOMAINS[r["Domain"]], r["Format"], r["Tag"]]}

    return items, convs, 512, score


def _gen_task(module, subject: str):
    """Task over the held-out verified problems in data/processed/gen_<subject>_test.jsonl."""
    def task(args):
        path = ROOT / "data" / "processed" / f"gen_{subject}_test.jsonl"
        if not path.exists():
            write_jsonl(module.generate(100, seed=2026, split="test"), path)
        items = _read_jsonl(path)
        shots_src = module.generate(args.fewshot, seed=999, split="train") if args.fewshot else []
        shots = [(s["messages"][1]["content"] + "\n\n" + GEN_INSTR, s["messages"][2]["content"]) for s in shots_src]
        convs = [conv(r["messages"][1]["content"] + "\n\n" + GEN_INSTR, shots) for r in items]

        def score(r, out):
            return grade(r["final"], out), {"group": [r["topic"].split("/")[0]]}

        return items, convs, 1024, score
    return task


task_gen_os = _gen_task(os_gen, "os")
task_gen_cn = _gen_task(cn_gen, "cn")
task_gen_dbms = _gen_task(dbms_gen, "dbms")
task_gen_dl = _gen_task(dl_gen, "dl")


def _run_python(code: str, timeout: int = 10) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        res = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def task_mbpp(args):
    from datasets import load_dataset
    test = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    items = random.Random(0).sample(test, min(args.mbpp_n, len(test)))
    prompt_split = list(load_dataset("google-research-datasets/mbpp", "full", split="prompt"))

    def text(r):
        return f"{r['text']}\nYour code should pass these tests:\n" + "\n".join(r["test_list"]) + f"\n\n{CODE_INSTR}"

    shots = [(text(r), f"```python\n{r['code']}\n```") for r in prompt_split[:args.fewshot]]
    convs = [conv(text(r), shots) for r in items]

    def score(r, out):
        code = extract_block(out, "python")
        program = "\n".join([r.get("test_setup_code") or "", code, *r["test_list"]])
        return _run_python(program), {}

    return items, convs, 512, score


def _exec_sql(db: Path, sql: str, timeout: float = 5.0):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    deadline = time.time() + timeout
    conn.set_progress_handler(lambda: int(time.time() > deadline), 10_000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _same_result(pred, gold, ordered: bool) -> bool:
    norm = lambda rows: [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in rows]
    p, g = norm(pred), norm(gold)
    return p == g if ordered else Counter(p) == Counter(g)


SPIDER_DIR = ROOT / "data" / "external" / "spider_data"
GATE_TEST = ROOT / "data" / "external" / "gate" / "gate_test.jsonl"


@functools.lru_cache(maxsize=None)
def spider_schema(db_path: Path) -> str:
    """CREATE TABLE statements of a Spider database (shared by evaluation and training data)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return "\n".join(s for (s,) in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"))
    finally:
        conn.close()


def spider_prompt(db_path: Path, question: str) -> str:
    return f"Database schema:\n{spider_schema(db_path)}\n\nQuestion: {question}\n\n{SQL_INSTR}"


def task_spider(args):
    sd = Path(args.spider_dir) if args.spider_dir else SPIDER_DIR
    if not (sd / "dev.json").exists():
        raise SystemExit(f"spider needs Spider at {sd} (dev.json + database/); see data/build_dataset.py")
    dev = json.loads((sd / "dev.json").read_text(encoding="utf-8"))
    items = random.Random(0).sample(dev, min(args.spider_n, len(dev)))

    def db_path(r):
        return sd / "database" / r["db_id"] / f"{r['db_id']}.sqlite"

    convs = [conv(spider_prompt(db_path(r), r["question"]), []) for r in items]

    def score(r, out):
        sql = extract_block(out, "sql")
        try:
            gold = _exec_sql(db_path(r), r["query"])
            pred = _exec_sql(db_path(r), sql)
        except Exception as e:  # invalid SQL, timeout, ...
            return False, {"sql": sql, "error": str(e)[:200]}
        return _same_result(pred, gold, "order by" in r["query"].lower()), {"sql": sql}

    return items, convs, 256, score


def task_gate(args):
    path = Path(args.gate_file) if args.gate_file else GATE_TEST
    if not path.exists():
        raise SystemExit(f"gate needs {path}: run `python data/build_gate.py` first")
    items = _read_jsonl(path)
    convs = [conv(_mcq_text({"Question": r["question"], **{k: r[k] for k in "ABCD"}}), []) for r in items]

    def score(r, out):
        pred = extract_choice(out)
        return pred == r["answer"].strip().upper(), {"pred": pred, "group": [r.get("subject", "all")]}

    return items, convs, 512, score


TASKS = {"csbench": task_csbench, "gen_os": task_gen_os, "gen_cn": task_gen_cn, "gen_dbms": task_gen_dbms, "gen_dl": task_gen_dl,
         "mbpp": task_mbpp, "spider": task_spider, "gate": task_gate}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def bootstrap_ci(xs: list[int], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def summarize(correct: list[int], groups: list[list[str]]) -> dict:
    lo, hi = bootstrap_ci(correct)
    out = {"n": len(correct), "acc": sum(correct) / len(correct), "ci95": [lo, hi], "by_group": {}}
    by = defaultdict(list)
    for c, gs in zip(correct, groups):
        for g in gs:
            by[g].append(c)
    out["by_group"] = {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(by.items())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", help="path to a LoRA adapter (fine-tuned runs)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tasks", nargs="+", default=["csbench", "gen_os", "mbpp"], choices=list(TASKS))
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, help="only first N items per task (smoke test)")
    ap.add_argument("--csbench-n", type=int, default=400)
    ap.add_argument("--mbpp-n", type=int, default=200)
    ap.add_argument("--spider-n", type=int, default=200)
    ap.add_argument("--spider-dir")
    ap.add_argument("--gate-file")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score outputs already saved in eval/results/<run>/ without running the model")
    args = ap.parse_args()

    out_dir = ROOT / "eval" / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = None if args.rescore else Generator(args.model, args.adapter, args.batch_size)
    summary_json = out_dir / "summary.json"
    meta = {"model": args.model, "adapter": args.adapter, "fewshot": args.fewshot}
    # Merge into an existing summary so tasks can be run (or re-run) separately.
    report = json.loads(summary_json.read_text(encoding="utf-8"))["results"] if summary_json.exists() else {}

    for name in args.tasks:
        print(f"[{args.run_name}] task {name}", flush=True)
        items, convs, max_new, score = TASKS[name](args)
        if args.limit:
            items, convs = items[:args.limit], convs[:args.limit]
        t0 = time.time()
        if args.rescore:
            saved = _read_jsonl(out_dir / f"{name}.jsonl")
            if len(saved) != len(items):
                raise SystemExit(f"{name}: {len(saved)} saved outputs but {len(items)} items; use the same --limit")
            outs = [s["output"] for s in saved]
        else:
            outs = gen.chat(convs, max_new)
        correct, groups = [], []
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item, c, out in zip(items, convs, outs):
                ok, info = score(item, out)
                correct.append(int(ok))
                groups.append(info.pop("group", []))
                f.write(json.dumps({"prompt": c[-1]["content"], "output": out, "correct": ok, **info},
                                   ensure_ascii=False, default=str) + "\n")
        seconds = report.get(name, {}).get("seconds") if args.rescore else round(time.time() - t0, 1)
        report[name] = {**summarize(correct, groups), "seconds": seconds}
        r = report[name]
        print(f"  {name}: acc={r['acc']:.3f}  95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  n={r['n']}")
        # Persist after every task so a crash later in the run keeps finished results.
        summary_json.write_text(json.dumps({"meta": meta, "results": report}, indent=2), encoding="utf-8")
        update_summary_csv(args.run_name, name, args.fewshot, r)

    print(json.dumps({k: round(v["acc"], 4) for k, v in report.items()}, indent=2))


def update_summary_csv(run: str, task: str, fewshot: int, r: dict) -> None:
    """Insert or replace the (run, task) row in eval/results/summary.csv."""
    path = ROOT / "eval" / "results" / "summary.csv"
    header = ["run", "task", "fewshot", "n", "acc", "ci_low", "ci_high"]
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [row for row in csv.reader(f)][1:]
    new = [run, task, str(fewshot), str(r["n"]), f"{r['acc']:.4f}", f"{r['ci95'][0]:.4f}", f"{r['ci95'][1]:.4f}"]
    idx = next((i for i, row in enumerate(rows) if row[:2] == [run, task]), None)
    if idx is None:
        rows.append(new)
    else:
        rows[idx] = new  # keep the row's position
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()
