"""Explanation-quality evaluation: blind pairwise comparison of two models' answers, judged by Gemini.

Final-answer accuracy (run_eval.py) does not say whether an answer *explains* well. This script asks
40 held-out questions (20 verified numericals with reference solutions, 20 concept questions of the kind
students type into the chat app) to two models through the chat server, then has Gemini compare the two
answers blind: the model names are hidden and every pair is judged twice with the order swapped. A model
wins a question only when it wins in both orders; otherwise the question counts as a tie.

  python eval/explain_eval.py answer --model base          # needs demo/server.py running
  python eval/explain_eval.py answer --model ft-c
  python eval/explain_eval.py judge --a base --b ft-c
  python eval/explain_eval.py report --a base --b ft-c

Answers are shown to the judge exactly as the chat app shows them (LaTeX converted to plain text).
Outputs go to eval/results/explain/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generators.grading import grade  # noqa: E402
from data.plain_math import has_latex, plain_math  # noqa: E402

OUT = ROOT / "eval" / "results" / "explain"
QUESTIONS = OUT / "questions.jsonl"
NUMERICAL_PER_SUBJECT = 5

# Held out: these questions (and near-duplicates) are excluded from all training data (see build_explain_stage.py).
CONCEPT_QUESTIONS = [
    ("DSA", "What is a binary search tree?"),
    ("DSA", "Explain why the worst-case time of quicksort is O(n^2) and how choosing a random pivot helps."),
    ("DSA", "What is the difference between BFS and DFS, and when would you use each?"),
    ("Programming", "What is recursion? Explain it with the factorial example."),
    ("Programming", "What is the difference between pass by value and pass by reference?"),
    ("Programming", "Write a Python function that checks whether a string is a palindrome and explain how it works."),
    ("DBMS", "What is normalization in DBMS?"),
    ("DBMS", "Explain the ACID properties of a transaction."),
    ("DBMS", "What is the difference between INNER JOIN and LEFT JOIN? Give an example."),
    ("OS", "What is a deadlock?"),
    ("OS", "What is thrashing and how can it be prevented?"),
    ("OS", "What is the difference between a process and a thread?"),
    ("OS", "What is virtual memory?"),
    ("CN", "Explain the TCP three-way handshake."),
    ("CN", "What is the difference between TCP and UDP?"),
    ("CN", "How does DNS work?"),
    ("DL", "What is CNN?"),
    ("DL", "What is the vanishing gradient problem?"),
    ("DL", "What is dropout and why does it help?"),
    ("DL", "What is the attention mechanism in transformers?"),
]

JUDGE_PROMPT = """You are an experienced Computer Science lecturer comparing two tutor answers to a student's question.
The student is an undergraduate; the ideal answer is correct, explains the steps or ideas in a logical order,
and is easy for a beginner to understand.

Question:
<<<
{question}
>>>
{reference}
Answer A:
<<<
{a}
>>>

Answer B:
<<<
{b}
>>>

Score each answer from 1 (poor) to 5 (excellent) on:
- correctness: are all facts, steps and the final result correct? {correctness_note}
- steps: is the reasoning or solution complete, in a logical order, with each step justified?
- clarity: would a beginner understand it (plain language, good structure, an example where useful)?
Then decide which answer is better overall for the student. Correctness matters most, then steps, then clarity.
Do not prefer an answer just because it is longer. Use "tie" only when they are genuinely equally good.

Return JSON only:
{{"A": {{"correctness": n, "steps": n, "clarity": n}}, "B": {{"correctness": n, "steps": n, "clarity": n}},
  "winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def build_questions() -> list[dict]:
    """20 verified numericals (one per distinct problem type, 5 per subject) + 20 concept questions."""
    qs = []
    for subject in ("os", "cn", "dbms", "dl"):
        seen = set()
        for r in read_jsonl(ROOT / "data" / "processed" / f"gen_{subject}_test.jsonl"):
            family = r["topic"]
            if family in seen:
                continue
            seen.add(family)
            qs.append({"id": r["id"], "kind": "numerical", "subject": r["subject"], "topic": r["topic"],
                       "question": r["messages"][1]["content"], "reference": r["messages"][2]["content"],
                       "final": r["final"]})
            if len(seen) == NUMERICAL_PER_SUBJECT:
                break
    for k, (subject, q) in enumerate(CONCEPT_QUESTIONS):
        qs.append({"id": f"concept-{k:02d}", "kind": "concept", "subject": subject, "question": q})
    return qs


def questions() -> list[dict]:
    if not QUESTIONS.exists():
        write_jsonl(build_questions(), QUESTIONS)
    return read_jsonl(QUESTIONS)


def ask(server: str, model: str, question: str, max_new_tokens: int) -> tuple[str, float]:
    body = json.dumps({"model": model, "temperature": 0.0, "max_new_tokens": max_new_tokens,
                       "messages": [{"role": "user", "content": question}]}).encode()
    req = urllib.request.Request(f"{server}/api/chat", data=body, headers={"Content-Type": "application/json"})
    text, seconds = "", 0.0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            text += ev.get("delta", "")
            seconds = ev.get("seconds", seconds)
    return text, seconds


def cmd_answer(args) -> None:
    path = OUT / f"answers_{args.model}.jsonl"
    done = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
    qs = questions()
    if not args.via_server:  # batched local generation: much faster than one answer at a time
        from eval.run_eval import Generator, conv
        adapter = None if args.model == "base" else str(ROOT / "outputs" / args.model / "adapter")
        gen = Generator("Qwen/Qwen2.5-1.5B-Instruct", adapter, args.batch_size)
        todo = [q for q in qs if q["id"] not in done]
        t0 = time.time()
        outs = gen.chat([conv(q["question"], []) for q in todo], args.max_new_tokens)
        for q, text in zip(todo, outs):
            row = {"id": q["id"], "model": args.model, "answer": text, "seconds": None}
            if q["kind"] == "numerical":
                row["final_correct"] = grade(q["final"], text)
            done[q["id"]] = row
        write_jsonl([done[x["id"]] for x in qs if x["id"] in done], path)
        print(f"{args.model}: {len(todo)} answers in {time.time() - t0:.0f}s", flush=True)
        return
    for _ in range(120):  # wait up to 10 min for the server to load the model
        try:
            urllib.request.urlopen(f"{args.server}/api/models", timeout=5)
            break
        except OSError:
            time.sleep(5)
    for k, q in enumerate(qs, 1):
        if q["id"] in done:
            continue
        text, seconds = ask(args.server, args.model, q["question"], args.max_new_tokens)
        row = {"id": q["id"], "model": args.model, "answer": text, "seconds": seconds}
        if q["kind"] == "numerical":
            row["final_correct"] = grade(q["final"], text)
        done[q["id"]] = row
        write_jsonl([done[x["id"]] for x in qs if x["id"] in done], path)
        print(f"[{k}/{len(qs)}] {args.model} {q['id']} {seconds:.0f}s {len(text.split())} words", flush=True)


def cmd_judge(args) -> None:
    from data.build_concepts import DEFAULT_MODELS, Gemini
    gem = Gemini(DEFAULT_MODELS, rpm=args.rpm)
    ans = {m: {r["id"]: r for r in read_jsonl(OUT / f"answers_{m}.jsonl")} for m in (args.a, args.b)}
    path = OUT / f"judge_{args.a}_vs_{args.b}.jsonl"
    done = {(r["id"], r["order"]): r for r in read_jsonl(path)} if path.exists() else {}
    qs = questions()
    for k, q in enumerate(qs, 1):
        for order in ("ab", "ba"):
            if (q["id"], order) in done or q["id"] not in ans[args.a] or q["id"] not in ans[args.b]:
                continue
            first, second = (args.a, args.b) if order == "ab" else (args.b, args.a)
            ref = (f"\nReference solution (verified correct; use it to check correctness):\n<<<\n{q['reference']}\n>>>\n"
                   if q["kind"] == "numerical" else "")
            note = ("If the final answer differs from the reference, correctness is at most 2."
                    if q["kind"] == "numerical" else "")
            prompt = JUDGE_PROMPT.format(question=q["question"], reference=ref, correctness_note=note,
                                         a=plain_math(ans[first][q["id"]]["answer"]),
                                         b=plain_math(ans[second][q["id"]]["answer"]))
            v = gem.generate_json(prompt, temperature=0.0)
            if isinstance(v, list):
                v = v[0]
            winner = {"A": first, "B": second}.get(v.get("winner"), "tie")
            done[(q["id"], order)] = {"id": q["id"], "order": order, "judge": gem.version, "winner": winner,
                                      "scores": {first: v.get("A"), second: v.get("B")}, "reason": v.get("reason")}
            write_jsonl(list(done.values()), path)
            print(f"[{k}/{len(qs)}] {q['id']} {order}: {winner}", flush=True)


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def cmd_report(args) -> dict:
    qs = {q["id"]: q for q in questions()}
    judged = read_jsonl(OUT / f"judge_{args.a}_vs_{args.b}.jsonl")
    ans = {m: {r["id"]: r for r in read_jsonl(OUT / f"answers_{m}.jsonl")} for m in (args.a, args.b)}
    by_q: dict[str, list[dict]] = {}
    for r in judged:
        by_q.setdefault(r["id"], []).append(r)
    res = {"a": args.a, "b": args.b, "n": 0, "groups": {}}
    for group in ("all", "numerical", "concept"):
        w = l = t = 0
        scores = {m: {"correctness": [], "steps": [], "clarity": []} for m in (args.a, args.b)}
        for qid, rs in by_q.items():
            if len(rs) < 2 or (group != "all" and qs[qid]["kind"] != group):
                continue
            winners = {r["winner"] for r in rs}
            if winners == {args.b}:
                w += 1
            elif winners == {args.a}:
                l += 1
            else:
                t += 1
            for r in rs:
                for m in (args.a, args.b):
                    for key in scores[m]:
                        if isinstance(r["scores"].get(m), dict) and r["scores"][m].get(key) is not None:
                            scores[m][key].append(float(r["scores"][m][key]))
        n = w + l + t
        res["groups"][group] = {
            "n": n, f"{args.b}_wins": w, f"{args.a}_wins": l, "ties": t, "p_sign": round(sign_test(w, l), 4),
            "mean_scores": {m: {k: round(sum(v) / len(v), 2) if v else None for k, v in s.items()}
                            for m, s in scores.items()},
        }
    for m in (args.a, args.b):
        rows = [ans[m][q] for q in qs if q in ans[m]]
        num = [r for r in rows if "final_correct" in r]
        res[m] = {"numerical_correct": sum(r["final_correct"] for r in num), "numerical_n": len(num),
                  "median_words": sorted(len(r["answer"].split()) for r in rows)[len(rows) // 2] if rows else None,
                  "answers_with_latex": sum(has_latex(r["answer"]) for r in rows)}
    res["n"] = res["groups"]["all"]["n"]
    (OUT / f"report_{args.a}_vs_{args.b}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answer")
    a.add_argument("--model", required=True)
    a.add_argument("--server", default="http://127.0.0.1:8000")
    a.add_argument("--via-server", action="store_true", help="ask the running chat server instead of batching locally")
    a.add_argument("--batch-size", type=int, default=16)
    a.add_argument("--max-new-tokens", type=int, default=1024)
    j = sub.add_parser("judge")
    j.add_argument("--a", required=True)
    j.add_argument("--b", required=True)
    j.add_argument("--rpm", type=float, default=8)
    r = sub.add_parser("report")
    r.add_argument("--a", required=True)
    r.add_argument("--b", required=True)
    args = ap.parse_args()
    {"answer": cmd_answer, "judge": cmd_judge, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
