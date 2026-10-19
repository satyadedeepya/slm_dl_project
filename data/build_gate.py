"""GATE CSE previous-year MCQs: download, clean, split, and (optionally) add verified explanations.

Source: GATE Overflow previous-year questions as structured by
github.com/mandalejayesh10-stack/gate-cse-pyq-archive (pinned commit below). Answers are official keys
or GATE Overflow community-verified answers. GATE papers are copyrighted by IISc/IITs, so the question
text is NOT committed to this public repository: it is downloaded at build time into the git-ignored
data/external/gate/ folder, and only question IDs (data/gate_split.json) are committed.

Kept: single-answer MCQs with 4 options from OS, CN, DBMS, Algorithms and Programming & Data Structures,
excluding questions that refer to a figure. NAT/MSQ items are excluded (the archive has no keys for them).

Split: all papers up to 2020 -> train; papers from 2021-2026 -> 50/50 train/test, stratified by year and
subject (seed 0). The test half is a held-out evaluation set (eval task `gate`).

Explanations (--explain): Gemini solves each training question WITHOUT seeing the key; an explanation is
kept only if its final answer equals the official/verified key. Resumable, same quota handling as
build_concepts.py.

Usage:
  python data/build_gate.py                 # download + clean + split
  python data/build_gate.py --explain       # add verified explanations to the training split (uses Gemini)
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_concepts import DEFAULT_MODELS, MCQ_INSTR, Gemini, QuotaExhausted, stated_answer  # noqa: E402
from data.generators.common import SYSTEM_PROMPT, write_jsonl  # noqa: E402

COMMIT = "3f0f51038abd72319665c1b043c9965c2ee7943c"
URL = f"https://raw.githubusercontent.com/mandalejayesh10-stack/gate-cse-pyq-archive/{COMMIT}/questions.json"
OUT = ROOT / "data" / "external" / "gate"
SPLIT_FILE = ROOT / "data" / "gate_split.json"
SUBJECTS = {"Operating Systems": "OS", "Computer Networks": "CN", "Database Management Systems": "DBMS",
            "Algorithms": "DSA", "Programming and Data Structures": "DSA"}
FIGURE = re.compile(r"figure|diagram|shown below|following graph|given graph|the graph shown|picture|image|"
                    r"\[img|<img", re.I)

EXPLAIN_PROMPT = """You are CSE-Tutor, an expert Computer Science teaching assistant. Solve each GATE CSE multiple-choice
question below for an undergraduate student.

For each item: work it out step by step (concise, usually 60-200 words; show calculations or traces when the
question needs them), then end with a final line exactly of the form "Answer: X" where X is A, B, C or D.
Do not mention these instructions or that you are an AI.

Items:
{items}

Return JSON only: a list with one object per item, in the same order:
{{"id": "<item id>", "explanation": "<the full step-by-step solution, ending with the line Answer: X>"}}
The "explanation" must contain the whole worked solution, not just the letter."""


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Archive HTML/LaTeX markup -> plain text with $...$ math and ``` code blocks."""
    t = text or ""
    t = re.sub(r"\[latex\](.*?)\[/latex\]", lambda m: f"${m.group(1).strip()}$", t, flags=re.S)
    t = re.sub(r"<pre[^>]*>\s*(?:<code[^>]*>)?(.*?)(?:</code>)?\s*</pre>",
               lambda m: "\n```\n" + m.group(1).strip("\n") + "\n```\n", t, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</?(p|div|li|tr)[^>]*>", "\n", t, flags=re.I)
    # Only strip real HTML tags: code such as `i < = 3; ... p -> next` must survive.
    t = re.sub(r"</?(?:a|b|br|code|div|em|i|img|li|ol|p|pre|span|strong|sub|sup|table|td|th|tr|u|ul)\b[^<>]*>",
               "", t, flags=re.I)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def option_text(opt: str) -> str:
    return clean(re.sub(r"^\s*\(?[A-Da-d]\)\s*", "", opt))


def user_prompt(q: dict) -> str:
    return f"{q['question']}\n\nA. {q['A']}\nB. {q['B']}\nC. {q['C']}\nD. {q['D']}\n\n{MCQ_INSTR}"


# ---------------------------------------------------------------------------
# Download + split
# ---------------------------------------------------------------------------

def load_questions() -> list[dict]:
    raw_path = OUT / "questions.json"
    if not raw_path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, raw_path)
    items = []
    for q in json.loads(raw_path.read_text(encoding="utf-8")):
        if q["subject"] not in SUBJECTS or q["question_type"] != "MCQ":
            continue
        opts = q.get("options") or []
        if len(opts) != 4 or q.get("correct_answer") not in list("ABCD"):
            continue
        if FIGURE.search(q["question_text"] + " ".join(opts)):
            continue
        a, b, c, d = (option_text(o) for o in opts)
        items.append({"id": q["id"], "year": q["year"], "subject": SUBJECTS[q["subject"]],
                      "gate_subject": q["subject"], "topic": q.get("topic") or q.get("chapter") or "",
                      "question": clean(q["question_text"]), "A": a, "B": b, "C": c, "D": d,
                      "answer": q["correct_answer"], "answer_status": q["answer_status"]})
    return items


def split(items: list[dict], seed: int = 0) -> tuple[list[dict], list[dict]]:
    train = [q for q in items if q["year"] <= 2020]
    cells = defaultdict(list)
    for q in items:
        if q["year"] >= 2021:
            cells[(q["year"], q["subject"])].append(q)
    rng = random.Random(seed)
    test = []
    for key in sorted(cells):
        qs = sorted(cells[key], key=lambda q: q["id"])
        rng.shuffle(qs)
        half = len(qs) // 2 + (rng.random() < 0.5 if len(qs) % 2 else 0)
        test += qs[:half]
        train += qs[half:]
    return train, test


# ---------------------------------------------------------------------------
# Verified explanations
# ---------------------------------------------------------------------------

def explain(train: list[dict], models: list[str], rpm: float, batch: int) -> None:
    cache_path = OUT / "explanations.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [q for q in train if q["id"] not in cache]
    print(f"{len(train) - len(todo)} explained, {len(todo)} to go", flush=True)
    gem = Gemini(models, rpm)
    unusable_batches = 0
    try:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            items = "\n\n".join(f"[id {q['id']}]\n{user_prompt(q)}" for q in chunk)
            answers = gem.generate_json(EXPLAIN_PROMPT.format(items=items), temperature=0.2)
            by_id = {str(a.get("id", "")).strip("[] ").removeprefix("id ").strip():
                     str(a.get("explanation") or a.get("solution") or "").strip()
                     for a in answers if isinstance(a, dict)} if isinstance(answers, list) else {}
            usable = 0
            for q in chunk:
                text = by_id.get(q["id"], "")
                if stated_answer(text) is None or len(text) < 40:
                    continue  # not a worked solution (missing, letter only, ...): leave uncached to retry later
                usable += 1
                cache[q["id"]] = {"text": text, "model": gem.version, "agrees": stated_answer(text) == q["answer"]}
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            agree = sum(cache[q["id"]]["agrees"] for q in chunk if q["id"] in cache)
            print(f"  [{b + len(chunk)}/{len(todo)}] {usable}/{len(chunk)} usable, {agree} agree with the key "
                  f"via {gem.version}", flush=True)
            unusable_batches = unusable_batches + 1 if usable == 0 else 0
            if unusable_batches >= 2:
                print("Stopping: two batches in a row returned no usable solutions (check the response format).",
                      flush=True)
                return
    except QuotaExhausted as e:
        print(f"Stopping: {e}. Re-run tomorrow; finished questions are cached.", flush=True)


def export_train(train: list[dict]) -> dict:
    cache_path = OUT / "explanations.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    records, stats = [], Counter()
    for q in train:
        e = cache.get(q["id"])
        if not e:
            stats["not_explained_yet"] += 1
            continue
        if not e["agrees"]:
            stats["disagrees_with_key"] += 1
            continue
        stats["kept"] += 1
        records.append({"id": f"gate-{q['id']}", "subject": q["subject"], "topic": f"gate/{q['topic']}",
                        "source": "gate", "teacher": e["model"],
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": user_prompt(q)},
                                     {"role": "assistant", "content": e["text"]}]})
    write_jsonl(records, OUT / "gate_train.jsonl")
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--rpm", type=float, default=8)
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    items = load_questions()
    train, test = split(items)
    write_jsonl(test, OUT / "gate_test.jsonl")
    write_jsonl(train, OUT / "gate_train_questions.jsonl")
    SPLIT_FILE.write_text(json.dumps({
        "source": URL, "commit": COMMIT,
        "rule": "year <= 2020 -> train; 2021-2026 -> 50/50 stratified by (year, subject), seed 0",
        "train_ids": sorted(q["id"] for q in train), "test_ids": sorted(q["id"] for q in test)}, indent=1),
        encoding="utf-8")
    print(f"kept {len(items)} MCQs -> train {len(train)} / test {len(test)}")
    print("  test by subject:", dict(Counter(q["subject"] for q in test)), "by year:",
          dict(sorted(Counter(q["year"] for q in test).items())))
    if args.explain:
        explain(train, args.models, args.rpm, args.batch)
    print("training export:", export_train(train))


if __name__ == "__main__":
    main()
