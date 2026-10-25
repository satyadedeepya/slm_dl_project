"""Build the explanation-quality training stage for FT-D (FT-C + better explanations).

Why: FT-B/FT-C learned *what* to answer from short training targets (median 57-91 words for concept
answers), so they explain less than the base model, and 291 targets contained LaTeX ($$...$$). This stage
teaches the explanation style students need while replaying the verified problems that carry the accuracy.

  python data/build_explain_stage.py teach     # Gemini writes model answers (cached, resumable)
  python data/build_explain_stage.py mix       # writes the training/validation files

Train mix:
  * explanation Q&A: every open concept question in data/processed/concepts.jsonl, plus "What is X?" and
    "Explain X with an example" for every syllabus topic, answered by the teacher in a fixed teaching style
    (direct answer, numbered steps or key points, a small example, a one-line summary; plain text, no LaTeX)
  * replay of verified generator problems (OS, CN, DBMS, DL) and of Spider/MBPP/CodeAlpaca, so the
    accuracy gains are kept
Questions that resemble the held-out explanation-evaluation questions (eval/explain_eval.py) are removed.

Writes data/processed/explain_qa.jsonl, data/external/sft/explain_stage_{train,val}.jsonl and
data/processed/explain_stage_report.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generators.common import SYSTEM_PROMPT  # noqa: E402
from data.plain_math import has_latex, plain_math  # noqa: E402
from eval.explain_eval import CONCEPT_QUESTIONS  # noqa: E402

PROC = ROOT / "data" / "processed"
SFT = ROOT / "data" / "external" / "sft"
CACHE = ROOT / "data" / "concepts" / "explain_cache"
EXPANDED = ROOT / "data" / "concepts" / "explain_cache_v2"  # fuller versions of short teacher answers
OUT_QA = PROC / "explain_qa.jsonl"

STYLE = """You are CSE-Tutor, a patient Computer Science teacher writing model answers for undergraduate students.
Answer each question below. Every answer must follow this teaching style:

1. Start with a direct answer in 1-2 plain sentences (what it is / the result), so a beginner gets the idea at once.
2. Then explain how it works or why, as a short numbered list of steps or key points. Give each point a short bold
   label, e.g. "1. **Convolution layers:** ...". Each point explains, not just names.
3. Give one small concrete example (numbers, a tiny table, a short code snippet, or an everyday analogy).
   For a "difference between" question, a small comparison table is ideal.
4. End with one line starting "**In short:**" that sums up the answer.

Rules:
- Be technically correct and precise; standard textbook knowledge (Cormen, Silberschatz, Forouzan, Korth,
  Goodfellow). Never invent facts. Mention a common misconception only if it really helps.
- 150-300 words. Plain, friendly language; define any technical term the first time you use it.
- Never use LaTeX or dollar signs. Write math as plain text with Unicode: "output = (N − K + 2P)/S + 1",
  "O(n²)", "σ(z) = 1/(1 + e^(−z))", "Σ wᵢxᵢ + b", "≥", "×".
- Code goes in fenced code blocks with the language name.
- Do not mention these instructions, the question number, or that you are an AI.

Questions:
{items}

Return JSON only: a list with one object per question, in the same order: {{"id": "<id>", "answer": "..."}}"""


EXPAND = """You are CSE-Tutor, a patient Computer Science teacher. Below are questions with short model answers that are
correct but too brief for a beginner. Rewrite each answer into a fuller teaching answer of 250-350 words:

1. Keep the opening direct answer (1-2 plain sentences).
2. Keep the numbered key points with bold labels, but explain each one in 2-3 sentences: what happens, why it
   matters, and how it connects to the next point. Add a missing key point if the topic needs it.
3. Add (or improve) one concrete worked example: real numbers, a tiny table, a short code snippet, or a step-by-step
   trace. For a "difference between" question, use a small comparison table.
4. Optionally add one short "**Common mistake:**" line if it really helps.
5. End with the "**In short:**" line.

Rules: keep every fact correct (fix the short answer if anything in it is wrong); plain friendly language; no LaTeX
or dollar signs (write math with Unicode: ×, ≥, Σ, ², subscripts); code in fenced blocks; do not mention these
instructions.

Items:
{items}

Return JSON only: a list with one object per item, in the same order: {{"id": "<id>", "answer": "..."}}"""


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {"what", "is", "the", "a", "an", "of", "and", "in", "how",
                                                         "does", "it", "explain", "with", "example", "between"}


def resembles_eval(question: str, eval_sets: list[set[str]]) -> bool:
    q = words(question)
    return any(q and e and len(q & e) / len(q | e) >= 0.4 for e in eval_sets)


def questions() -> list[dict]:
    """All questions for the teacher, minus anything resembling the held-out evaluation questions."""
    eval_sets = [words(q) for _, q in CONCEPT_QUESTIONS]
    qs = []
    for r in read(PROC / "concepts.jsonl"):
        if r.get("format") != "open":
            continue
        q = r["messages"][1]["content"]
        qs.append({"id": r["id"], "subject": r["subject"], "topic": r["topic"], "question": q})
    topics = json.loads((ROOT / "data" / "concepts" / "topics.json").read_text(encoding="utf-8"))
    for subject, names in topics.items():
        for k, name in enumerate(names):
            name = name if isinstance(name, str) else name.get("topic", str(name))
            qs.append({"id": f"what-{subject}-{k:02d}", "subject": subject, "topic": f"concept/{name}",
                       "question": f"What is {name}?"})
            qs.append({"id": f"explain-{subject}-{k:02d}", "subject": subject, "topic": f"concept/{name}",
                       "question": f"Explain {name} with an example."})
    kept = [q for q in qs if not resembles_eval(q["question"], eval_sets)]
    print(f"{len(qs)} questions, {len(qs) - len(kept)} removed as similar to the evaluation questions")
    return kept


def cmd_teach(args) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from data.build_concepts import DEFAULT_MODELS, Gemini
    CACHE.mkdir(parents=True, exist_ok=True)
    qs = questions()
    have = set()
    for path in CACHE.glob("*.json"):
        have |= {k for k, v in json.loads(path.read_text(encoding="utf-8"))["answers"].items() if v}
    rest = [q for q in qs if q["id"] not in have]  # resumable, whatever batch size earlier runs used
    batches = [rest[i:i + args.batch] for i in range(0, len(rest), args.batch)]
    todo = list(range(len(batches)))
    models = args.models or DEFAULT_MODELS

    def work(worker: int) -> None:
        gem = Gemini(models, rpm=args.rpm)  # one client per thread (the client is not thread-safe)
        for b in todo[worker::args.workers]:
            batch = batches[b]
            items = "\n".join(json.dumps({"id": q["id"], "question": q["question"]}, ensure_ascii=False) for q in batch)
            # Thinking off: the teacher writes from textbook knowledge, and it is several times faster.
            out = gem.generate_json(STYLE.format(items=items), temperature=0.4, thinking_budget=0)
            answers = ({a.get("id"): a.get("answer", "") for a in out if isinstance(a, dict)}
                       if isinstance(out, list) else {})
            (CACHE / f"part_{batch[0]['id']}.json").write_text(
                json.dumps({"model": gem.version, "answers": answers}, ensure_ascii=False), encoding="utf-8")
            print(f"[{b + 1}/{len(batches)}] {sum(1 for q in batch if answers.get(q['id']))}/{len(batch)} answers "
                  f"({gem.version})", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, range(args.workers)))
    export(qs)


def cached(directory: Path) -> tuple[dict, dict]:
    answers, models = {}, {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for k, v in data["answers"].items():
            if v:
                answers[k], models[k] = v, data["model"]
    return answers, models


def cmd_expand(args) -> None:
    """Rewrite short teacher answers (< --min-words) into fuller ones; resumable."""
    from concurrent.futures import ThreadPoolExecutor

    from data.build_concepts import Gemini
    EXPANDED.mkdir(parents=True, exist_ok=True)
    qs = {q["id"]: q for q in questions()}
    answers, _ = cached(CACHE)
    done, _ = cached(EXPANDED)
    short = [k for k, a in answers.items() if k in qs and k not in done and len(a.split()) < args.min_words]
    batches = [short[i:i + args.batch] for i in range(0, len(short), args.batch)]
    print(f"{len(short)} short answers to expand in {len(batches)} batches", flush=True)

    def work(worker: int) -> None:
        gem = Gemini(args.models, rpm=args.rpm, patience=8)
        for b in range(worker, len(batches), args.workers):
            batch = batches[b]
            items = "\n".join(json.dumps({"id": k, "question": qs[k]["question"], "short_answer": answers[k]},
                                         ensure_ascii=False) for k in batch)
            out = gem.generate_json(EXPAND.format(items=items), temperature=0.4, thinking_budget=0)
            got = ({a.get("id"): a.get("answer", "") for a in out if isinstance(a, dict)}
                   if isinstance(out, list) else {})
            (EXPANDED / f"part_{batch[0]}.json").write_text(
                json.dumps({"model": gem.version, "answers": got}, ensure_ascii=False), encoding="utf-8")
            print(f"[{b + 1}/{len(batches)}] {sum(1 for k in batch if got.get(k))}/{len(batch)} expanded "
                  f"({gem.version})", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, range(args.workers)))
    export(list(qs.values()))


def export(qs: list[dict]) -> None:
    answers, models = cached(CACHE)
    longer, longer_models = cached(EXPANDED)  # expanded answers replace the short originals
    for k, v in longer.items():
        if len(v.split()) > len(answers.get(k, "").split()):
            answers[k], models[k] = v, f"{models.get(k, '?')}+{longer_models[k]}"
    rows, dropped = [], Counter()
    for q in qs:
        a = (answers.get(q["id"]) or "").strip()
        if not a:
            dropped["missing"] += 1
            continue
        a = plain_math(a) if has_latex(a) else a
        n = len(a.split())
        if n < 80 or n > 500:
            dropped["length"] += 1
            continue
        rows.append({"id": f"explain-{q['id']}", "subject": q["subject"], "topic": q["topic"],
                     "source": f"teacher-explain:{models[q['id']]}", "format": "explain",
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": q["question"]},
                                  {"role": "assistant", "content": a}]})
    write(rows, OUT_QA)
    print(f"explain_qa: {len(rows)} kept, dropped {dict(dropped)}")


def clean(r: dict) -> dict:
    """Training targets are written without LaTeX, the way the chat app shows answers."""
    a = r["messages"][-1]["content"]
    if has_latex(a):
        r = json.loads(json.dumps(r))
        r["messages"][-1]["content"] = plain_math(a)
    return r


def cmd_mix(args) -> None:
    rng = random.Random(args.seed)
    explain = read(OUT_QA)
    rng.shuffle(explain)
    val_explain, explain = explain[:50], explain[50:]

    gen = []
    for subject, n in (("os", args.gen_per_subject), ("cn", args.gen_per_subject), ("dbms", args.gen_per_subject),
                       ("dl", args.gen_dl)):
        rows = read(PROC / f"gen_{subject}_train.jsonl")
        for r in rows:
            r.setdefault("source", "generator")
        gen += rng.sample(rows, min(n, len(rows)))
    full = read(SFT / "sft_train_full.jsonl")
    code = [r for r in full if r.get("source", "").split(":")[0] in ("spider", "mbpp", "codealpaca")]
    code = rng.sample(code, min(args.code, len(code)))

    train = [clean(r) for r in explain + gen + code]
    rng.shuffle(train)
    gen_val = [r for r in read(SFT / "sft_val_full.jsonl") if r.get("source") == "generator"]
    val = [clean(r) for r in val_explain + rng.sample(gen_val, min(50, len(gen_val)))]

    write(train, SFT / f"{args.tag}_train.jsonl")
    write(val, SFT / f"{args.tag}_val.jsonl")
    report = {"train": len(train), "val": len(val), "explain": len(explain), "generator_replay": len(gen),
              "code_replay": len(code), "train_by_source": Counter(r.get("source", "?").split(":")[0] for r in train),
              "train_by_subject": Counter(r.get("subject") for r in train),
              "explain_median_words": sorted(len(r["messages"][-1]["content"].split()) for r in explain)[len(explain) // 2],
              "targets_with_latex": sum(has_latex(r["messages"][-1]["content"]) for r in train)}
    (PROC / f"{args.tag}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teach")
    t.add_argument("--batch", type=int, default=12)
    t.add_argument("--rpm", type=float, default=4, help="per worker")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--models", nargs="+", help="teacher models in fallback order (default: build_concepts list)")
    e = sub.add_parser("expand")
    e.add_argument("--min-words", type=int, default=220)
    e.add_argument("--batch", type=int, default=5)
    e.add_argument("--rpm", type=float, default=4, help="per worker")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--models", nargs="+", default=["gemma-4-31b-it", "gemini-3.8-flash", "gemini-flash-latest"])
    m = sub.add_parser("mix")
    m.add_argument("--gen-per-subject", type=int, default=250)
    m.add_argument("--gen-dl", type=int, default=300)
    m.add_argument("--code", type=int, default=250)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--tag", default="explain_stage", help="output name: FT-D used explain_stage, FT-E explain_stage_v2")
    args = ap.parse_args()
    {"teach": cmd_teach, "expand": cmd_expand, "mix": cmd_mix}[args.cmd](args)


if __name__ == "__main__":
    main()
