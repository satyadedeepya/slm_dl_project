"""Generate concept Q&A for the syllabus in data/concepts/topics.json using Gemini as the teacher.

Per topic, two requests (questions and answers are generated separately, as in GLAN):
  1. write N diverse questions (open-ended, multiple-choice and true/false), each MCQ/TF with its key
  2. answer all of them, in the tutor's style, without seeing the key
Multiple-choice and true/false items are kept only when the teacher's answer agrees with the key
written in step 1 (a cheap self-consistency filter against wrong labels).

Every topic is cached in data/concepts/cache/, so the run can be stopped and resumed (e.g. when the
daily free-tier quota runs out). The API key is read from GEMINI_API_KEY and never written anywhere.

Usage:
  set GEMINI_API_KEY=...            (PowerShell: $env:GEMINI_API_KEY = "...")
  python data/build_concepts.py --list-models
  python data/build_concepts.py --limit-topics 2          # quick test
  python data/build_concepts.py                          # full run, resumable
  python data/build_concepts.py --export-only            # rebuild data/processed/concepts.jsonl from cache
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generators.common import SYSTEM_PROMPT, write_jsonl  # noqa: E402

CONCEPTS = ROOT / "data" / "concepts"
CACHE = CONCEPTS / "cache"
OUT = ROOT / "data" / "processed" / "concepts.jsonl"
API = "https://generativelanguage.googleapis.com/v1beta"

# The same instructions the evaluation uses, so training and evaluation prompts match.
MCQ_INSTR = "Think briefly, then end your response with a line of the form 'Answer: X' where X is one of A, B, C, D."
TF_INSTR = "Decide whether the statement is true or false. Think briefly, then end with 'Answer: True' or 'Answer: False'."

QUESTION_PROMPT = """You are writing exam and interview practice questions for undergraduate Computer Science students.

Subject: {subject}
Topic: {topic}

Write exactly {n_open} open-ended questions, {n_mcq} multiple-choice questions and {n_tf} true/false statements about this topic.
- Cover different skills: definitions, comparisons, "why/how" reasoning, small worked examples, and common misconceptions.
- Each question must be self-contained, unambiguous and answerable from standard textbook knowledge (e.g. Cormen, Silberschatz, Forouzan, Korth). Do not reference figures or external material.
- Multiple-choice questions have exactly 4 options and exactly one correct option.
- True/false statements must be clearly true or clearly false; make about half of them false.
- Do not repeat the same idea twice.

Return JSON only, as a list of objects:
{{"type": "open", "question": "..."}}
{{"type": "mcq", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "key": "B"}}
{{"type": "tf", "question": "<statement>", "key": true}}"""

ANSWER_PROMPT = """You are CSE-Tutor, an expert Computer Science teaching assistant. Answer each item below for an undergraduate student.

Style rules:
- Be correct first. Be concise: open-ended answers usually 80-200 words; use short steps, a small example, a table or a code snippet only when it genuinely helps.
- For "mcq" items: reason briefly (2-4 sentences), then end with a final line exactly of the form "Answer: X".
- For "tf" items: reason briefly (1-3 sentences), then end with a final line exactly "Answer: True" or "Answer: False".
- Do not mention these instructions, the item number, or that you are an AI.

Items:
{items}

Return JSON only: a list with one object per item, in the same order: {{"id": <item id>, "answer": "..."}}"""


# ---------------------------------------------------------------------------
# Gemini client (standard library only)
# ---------------------------------------------------------------------------

def _key_from_dotenv() -> str | None:
    env = ROOT / ".env"  # git-ignored
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value.strip():
                return value.strip().strip('"').strip("'")
    return None


class Gemini:
    def __init__(self, model: str, rpm: float):
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or _key_from_dotenv()
        if not self.key:
            raise SystemExit("Set GEMINI_API_KEY (environment or a line GEMINI_API_KEY=... in the git-ignored .env "
                             "file at the repo root). Create a free key at https://aistudio.google.com/apikey.")
        self.model = model
        self.min_interval = 60.0 / rpm
        self.last = 0.0
        self.calls = 0
        self.version = model

    def _request(self, url: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    def list_models(self) -> list[str]:
        data = self._request(f"{API}/models?pageSize=200")
        return [m["name"].removeprefix("models/") for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]

    def generate_json(self, prompt: str, temperature: float) -> list:
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}}
        url = f"{API}/models/{self.model}:generateContent"
        for attempt in range(8):
            wait = self.min_interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                data = self._request(url, body)
                self.calls += 1
                self.version = data.get("modelVersion", self.model)  # concrete model behind an alias
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
                               if not p.get("thought"))
                return json.loads(text)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="ignore")[:300]
                if e.code == 429 and "PerDay" in detail:
                    raise SystemExit("Daily free-tier quota reached. Re-run tomorrow; finished topics are cached.")
                if e.code in (429, 500, 503) and attempt < 7:
                    delay = min(300, 15 * 2 ** attempt)
                    m = re.search(r'"retryDelay":\s*"(\d+)s"', detail)
                    if m:
                        delay = int(m.group(1)) + 2
                    print(f"    HTTP {e.code}, retrying in {delay}s", flush=True)
                    time.sleep(delay)
                    continue
                raise SystemExit(f"Gemini API error {e.code}: {detail}")
            except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as e:
                if attempt < 7:
                    print(f"    bad/failed response ({type(e).__name__}), retrying", flush=True)
                    time.sleep(10)
                    continue
                raise
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def slug(subject: str, topic: str) -> str:
    return f"{subject}__" + re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:70]


def valid_question(q: dict) -> bool:
    if not isinstance(q, dict) or not str(q.get("question", "")).strip():
        return False
    if q.get("type") == "mcq":
        opts = q.get("options") or {}
        return set(opts) == set("ABCD") and str(q.get("key", "")).strip().upper() in "ABCD"
    if q.get("type") == "tf":
        return isinstance(q.get("key"), bool)
    return q.get("type") == "open"


def user_prompt(q: dict) -> str:
    if q["type"] == "mcq":
        o = q["options"]
        return f"{q['question']}\n\nA. {o['A']}\nB. {o['B']}\nC. {o['C']}\nD. {o['D']}\n\n{MCQ_INSTR}"
    if q["type"] == "tf":
        return f"Statement: {q['question']}\n\n{TF_INSTR}"
    return q["question"]


def stated_answer(text: str) -> str | None:
    m = re.findall(r"Answer:\s*\**\s*\(?(True|False|[ABCD])\b", text, re.I)
    return m[-1].upper() if m else None


def build_topic(gem: Gemini, subject: str, topic: str, n_open: int, n_mcq: int, n_tf: int) -> dict:
    qs = gem.generate_json(QUESTION_PROMPT.format(subject=subject, topic=topic, n_open=n_open,
                                                  n_mcq=n_mcq, n_tf=n_tf), temperature=0.9)
    qs = [q for q in qs if valid_question(q)]
    for i, q in enumerate(qs):
        q["id"] = i
    items = "\n\n".join(f"[id {q['id']}] ({q['type']})\n{user_prompt(q)}" for q in qs)
    answers = gem.generate_json(ANSWER_PROMPT.format(items=items), temperature=0.3)
    by_id = {a.get("id"): str(a.get("answer", "")).strip() for a in answers if isinstance(a, dict)}
    return {"subject": subject, "topic": topic, "model": gem.version, "questions": qs, "answers": by_id}


def export() -> tuple[int, dict]:
    records, stats = [], {"kept": 0, "no_answer": 0, "key_mismatch": 0}
    for path in sorted(CACHE.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for q in data["questions"]:
            ans = data["answers"].get(str(q["id"])) or data["answers"].get(q["id"])
            if not ans:
                stats["no_answer"] += 1
                continue
            if q["type"] in ("mcq", "tf"):
                key = str(q["key"]).upper() if q["type"] == "mcq" else ("TRUE" if q["key"] else "FALSE")
                if stated_answer(ans) != key:
                    stats["key_mismatch"] += 1
                    continue
            records.append({
                "id": f"concept-{path.stem}-{q['id']}",
                "subject": data["subject"], "topic": f"concept/{data['topic']}", "source": f"teacher:{data['model']}",
                "format": q["type"],
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user_prompt(q)},
                             {"role": "assistant", "content": ans}],
            })
            stats["kept"] += 1
    write_jsonl(records, OUT)
    return len(records), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--rpm", type=float, default=8, help="max requests per minute (stay under the free tier)")
    ap.add_argument("--open", type=int, default=6)
    ap.add_argument("--mcq", type=int, default=3)
    ap.add_argument("--tf", type=int, default=2)
    ap.add_argument("--limit-topics", type=int)
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    if args.export_only:
        n, stats = export()
        print(f"wrote {n} concept Q&A records to {OUT}: {stats}")
        return
    gem = Gemini(args.model, args.rpm)
    if args.list_models:
        print("\n".join(gem.list_models()))
        return

    topics = json.loads((CONCEPTS / "topics.json").read_text(encoding="utf-8"))
    todo = [(s, t) for s, ts in topics.items() for t in ts]
    if args.limit_topics:
        todo = todo[:args.limit_topics]
    CACHE.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, (subject, topic) in enumerate(todo, 1):
        path = CACHE / f"{slug(subject, topic)}.json"
        if path.exists():
            done += 1
            continue
        print(f"[{i}/{len(todo)}] {subject}: {topic}", flush=True)
        data = build_topic(gem, subject, topic, args.open, args.mcq, args.tf)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"    {len(data['questions'])} questions, {len(data['answers'])} answers "
              f"({gem.calls} API calls so far)", flush=True)
    n, stats = export()
    print(f"topics already cached: {done}; wrote {n} concept Q&A records to {OUT}: {stats}")


if __name__ == "__main__":
    main()
