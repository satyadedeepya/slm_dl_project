"""Question-answer pairs grounded in open-licensed textbooks (Gemini as the writer).

Books (all openly licensed; attribution is kept on every record):
  CN    Peterson & Davie, Computer Networks: A Systems Approach        CC BY 4.0     (reStructuredText source)
  OS    Hailperin, Operating Systems and Middleware                     CC BY-SA 3.0  (LaTeX source; the
        transactions chapter is labelled DBMS)
  DSA   Erickson, Algorithms                                            CC BY 4.0     (PDF)
  DSA   Morin, Open Data Structures (Python edition)                    CC BY 2.5 CA  (PDF)

Pipeline:
  1. download each book into the git-ignored data/external/textbooks/
  2. extract plain text and split it into ~700-word passages
  3. pick up to --per-book passages spread evenly through each book (skipping front/back matter)
  4. ask Gemini for question-answer pairs that must be answerable from the passage alone
     (open-ended + one MCQ + one true/false per passage); passages that are only exercises,
     references or listings return nothing
  5. MCQ/TF items are kept only if the answer's final line matches the key

Resumable: each request's result is cached. Same quota handling as build_concepts.py.

Usage:
  python data/build_textbook_qa.py --prepare-only      # download + passages, no API calls
  python data/build_textbook_qa.py                     # generate (needs GEMINI_API_KEY)
  python data/build_textbook_qa.py --export-only       # rebuild data/processed/textbook_qa.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_concepts import (DEFAULT_MODELS, MCQ_INSTR, TF_INSTR, Gemini, QuotaExhausted,  # noqa: E402
                                 stated_answer)
from data.generators.common import SYSTEM_PROMPT, write_jsonl  # noqa: E402

BOOKS_DIR = ROOT / "data" / "external" / "textbooks"
CACHE = BOOKS_DIR / "qa_cache"
OUT = ROOT / "data" / "processed" / "textbook_qa.jsonl"

BOOKS = {
    "peterson_davie_cn": {
        "subject": "CN", "title": "Computer Networks: A Systems Approach", "authors": "Larry Peterson, Bruce Davie",
        "license": "CC BY 4.0", "url": "https://github.com/SystemsApproach/book",
        "kind": "rst", "fetch": "https://github.com/SystemsApproach/book/archive/refs/heads/master.zip",
        "chapters": ["foundation", "direct", "internetworking", "scaling", "e2e", "congestion", "data",
                     "security", "applications"],
    },
    "hailperin_os": {
        "subject": "OS", "title": "Operating Systems and Middleware: Supporting Controlled Interaction",
        "authors": "Max Hailperin", "license": "CC BY-SA 3.0",
        "url": "https://github.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction",
        "kind": "tex",
        "fetch": "https://github.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction"
                 "/archive/refs/heads/master.zip",
        "chapters": ["intro", "threads", "scheduling", "synchronization", "vm", "processes", "persistence",
                     "security", "transactions"],
        "chapter_subject": {"transactions": "DBMS"},
    },
    "erickson_algorithms": {
        "subject": "DSA", "title": "Algorithms", "authors": "Jeff Erickson", "license": "CC BY 4.0",
        "url": "https://jeffe.cs.illinois.edu/teaching/algorithms/", "kind": "pdf",
        "fetch": "https://jeffe.cs.illinois.edu/teaching/algorithms/book/Algorithms-JeffE.pdf",
    },
    "morin_ods": {
        "subject": "DSA", "title": "Open Data Structures (in Python)", "authors": "Pat Morin",
        "license": "CC BY 2.5 CA", "url": "https://opendatastructures.org/", "kind": "pdf",
        "fetch": "https://opendatastructures.org/ods-python.pdf",
    },
}

PROMPT = """You are writing study material for undergraduate Computer Science students from textbook passages.

For EACH passage below write {n_open} open-ended question-answer pairs, 1 multiple-choice question and 1 true/false
statement, ALL answerable from that passage alone.
- Questions must be self-contained: never refer to "the passage", "the text", "the author" or "this chapter".
- Answers must be correct and fully supported by the passage, written in your own words for a student, concise
  (usually 50-180 words), with a short example or calculation from the passage when it helps.
- Multiple-choice: exactly 4 options (A-D), one correct; the answer reasons briefly and ends with a final line
  exactly "Answer: X".
- True/false: a clearly true or clearly false statement; the answer reasons briefly and ends with a final line
  exactly "Answer: True" or "Answer: False".
- "answer" is always the full written answer text (for MCQ/TF: the reasoning followed by the Answer line), never
  just a letter or True/False.
- If a passage is mainly exercises, references, an index, front/back matter or code without explanation, return an
  empty list for it.

{passages}

Return JSON only: an object mapping each passage number (as a string) to a list of items:
{{"type": "open", "question": "...", "answer": "..."}}
{{"type": "mcq", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "key": "C", "answer": "..."}}
{{"type": "tf", "question": "<statement>", "key": false, "answer": "..."}}"""


# ---------------------------------------------------------------------------
# Download + text extraction
# ---------------------------------------------------------------------------

def fetch(book: str) -> Path:
    meta = BOOKS[book]
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    target = BOOKS_DIR / (book + (".pdf" if meta["kind"] == "pdf" else ".zip"))
    if not target.exists():
        req = urllib.request.Request(meta["fetch"], headers={"User-Agent": "Mozilla/5.0 (course project)"})
        with urllib.request.urlopen(req, timeout=300) as r:
            target.write_bytes(r.read())
    if meta["kind"] != "pdf":
        out = BOOKS_DIR / book
        if not out.exists():
            zipfile.ZipFile(target).extractall(out)
        return out
    return target


def rst_to_text(src: str) -> str:
    lines, out, skip_indent = src.splitlines(), [], False
    for line in lines:
        if line.startswith(".. "):  # directives (figures, images, includes, comments) and their bodies
            skip_indent = True
            continue
        if skip_indent and (line.startswith("   ") or not line.strip()):
            continue
        skip_indent = False
        if re.fullmatch(r"\s*([=\-~^*#\"'`+])\1{2,}\s*", line):  # heading underlines
            continue
        out.append(line)
    t = "\n".join(out)
    t = re.sub(r":\w+:`([^`<]*?)\s*(?:<[^>]*>)?`", r"\1", t)  # :role:`text <target>`
    t = re.sub(r"`([^`<]+?)\s*<[^>]+>`_+", r"\1", t)        # `text <url>`_
    t = re.sub(r"``([^`]+)``", r"\1", t)
    t = re.sub(r"\|[^|\s]+\|", "", t)                         # substitutions
    return t


def tex_to_text(src: str) -> str:
    from pylatexenc.latex2text import LatexNodes2Text
    src = re.sub(r"(?<!\\)%.*", "", src)  # comments
    src = re.sub(r"\\begin\{(figure|table|verbatim|lstlisting|tabular)\*?\}.*?\\end\{\1\*?\}", " ", src, flags=re.S)
    src = re.sub(r"\\(index|label|ref|pageref|cite|footnote)\{[^{}]*\}", "", src)
    src = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", src)
    conv = LatexNodes2Text(math_mode="verbatim")
    out = []
    for para in re.split(r"\n\s*\n", src):  # convert paragraph by paragraph so one bad macro cannot sink a chapter
        try:
            out.append(conv.latex_to_text(para))
        except Exception:
            t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", para)  # drop commands, keep their braced text
            out.append(re.sub(r"[{}]", "", t))
    return "\n\n".join(out)


OLDSTYLE = {name: str(i) for i, name in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])}


def fix_pdf_glyphs(text: str) -> str:
    """Repair glyph names that some PDFs emit for small caps / old-style digits, ligatures and hyphenation."""
    import unicodedata
    text = re.sub(r"/([A-Za-z])\.c2sc", r"\1", text)                                  # small caps of capitals
    text = re.sub(r"/([a-z])\.sc", lambda m: m.group(1).upper(), text)                 # small caps of lowercase
    text = re.sub(r"/(zero|one|two|three|four|five|six|seven|eight|nine)\.(?:Alt\.)?(?:oldstyle|osf|taboldstyle)",
                  lambda m: OLDSTYLE[m.group(1)], text)                                # old-style digits
    text = re.sub(r"/(period|comma|hyphen|colon)\.(?:c2sc|oldstyle)",
                  lambda m: {"period": ".", "comma": ",", "hyphen": "-", "colon": ":"}[m.group(1)], text)
    text = unicodedata.normalize("NFKC", text)                                         # ﬃ -> ffi, etc.
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)                                       # line-break hyphenation


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader
    pages = []
    for page in PdfReader(str(path)).pages:
        text = fix_pdf_glyphs(page.extract_text() or "")
        # drop running headers/footers that are just page numbers
        kept = [ln for ln in text.splitlines() if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
        pages.append("\n".join(kept))
    return "\n\n".join(pages)


EXERCISE_HINTS = re.compile(r"\bExercises?\b|Describe and analyze|\(\s*[a-e]\s*\)\s+(?:Describe|Prove|Show|Give)", re.I)


def is_exercise_heavy(text: str) -> bool:
    return len(EXERCISE_HINTS.findall(text)) >= 2


def book_sections(book: str) -> list[tuple[str, str, str]]:
    """(subject, section label, text) in reading order."""
    meta, path = BOOKS[book], fetch(book)
    if meta["kind"] == "pdf":
        return [(meta["subject"], meta["title"], pdf_to_text(path))]
    root = next(path.iterdir())
    out = []
    for ch in meta["chapters"]:
        subject = meta.get("chapter_subject", {}).get(ch, meta["subject"])
        if meta["kind"] == "rst":
            files = [root / f"{ch}.rst"] + sorted((root / ch).glob("*.rst"), key=lambda p: p.name)
            text = "\n\n".join(rst_to_text(f.read_text(encoding="utf-8")) for f in files if f.exists())
        else:
            text = tex_to_text((root / f"{ch}.tex").read_text(encoding="utf-8"))
        out.append((subject, ch, text))
    return out


def passages(book: str, words: int = 700) -> list[dict]:
    out = []
    for subject, label, text in book_sections(book):
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
        buf, n = [], 0
        for p in paras:
            if not p:
                continue
            buf.append(p)
            n += len(p.split())
            if n >= words:
                out.append({"book": book, "subject": subject, "section": label, "text": "\n\n".join(buf)})
                buf, n = [], 0
        if n >= words // 3:
            out.append({"book": book, "subject": subject, "section": label, "text": "\n\n".join(buf)})
    for i, p in enumerate(out):
        p["id"] = f"{book}-{i:04d}"
    return out


def select(ps: list[dict], k: int, skip_frac: float = 0.03, max_minor: int = 25) -> list[dict]:
    """Evenly spaced passages, skipping front/back matter and exercise-heavy passages.

    Passages of a book's minority subject (e.g. Hailperin's transactions chapter, labelled DBMS) are
    kept in full (up to `max_minor`) so the small subject is not sampled away.
    """
    lo, hi = int(len(ps) * skip_frac), len(ps) - int(len(ps) * skip_frac)
    core = [p for p in ps[lo:hi] if not is_exercise_heavy(p["text"])]
    main_subject = Counter(p["subject"] for p in core).most_common(1)[0][0]
    minor = [p for p in core if p["subject"] != main_subject][:max_minor]
    rest = [p for p in core if p["subject"] == main_subject]
    k_rest = max(0, k - len(minor))
    if len(rest) > k_rest:
        step = len(rest) / k_rest
        rest = [rest[int(i * step)] for i in range(k_rest)]
    return sorted(minor + rest, key=lambda p: p["id"])


# ---------------------------------------------------------------------------
# Generation + export
# ---------------------------------------------------------------------------

def user_prompt(item: dict) -> str:
    if item["type"] == "mcq":
        o = item["options"]
        return f"{item['question']}\n\nA. {o['A']}\nB. {o['B']}\nC. {o['C']}\nD. {o['D']}\n\n{MCQ_INSTR}"
    if item["type"] == "tf":
        return f"Statement: {item['question']}\n\n{TF_INSTR}"
    return item["question"]


def valid(item: dict) -> bool:
    if not isinstance(item, dict) or not str(item.get("question", "")).strip() or not str(item.get("answer", "")).strip():
        return False
    if item.get("type") == "mcq":
        return set(item.get("options") or {}) == set("ABCD") and str(item.get("key", "")).upper() in list("ABCD")
    if item.get("type") == "tf":
        return isinstance(item.get("key"), bool)
    return item.get("type") == "open"


def generate(selected: list[dict], models: list[str], rpm: float, per_call: int, n_open: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [p for p in selected if not (CACHE / f"{p['id']}.json").exists()]
    print(f"{len(selected) - len(todo)} passages done, {len(todo)} to go", flush=True)
    gem = Gemini(models, rpm)
    try:
        for b in range(0, len(todo), per_call):
            chunk = todo[b:b + per_call]
            text = "\n\n".join(f"### Passage {i}\n{p['text']}" for i, p in enumerate(chunk, 1))
            raw = gem.generate_json(PROMPT.format(n_open=n_open, passages=text), temperature=0.4)
            for i, p in enumerate(chunk, 1):
                items = raw.get(str(i), []) if isinstance(raw, dict) else []
                (CACHE / f"{p['id']}.json").write_text(json.dumps(
                    {"passage": p, "model": gem.version, "items": items}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
            print(f"  [{b + len(chunk)}/{len(todo)}] {sum(len(raw.get(str(i), [])) for i in range(1, len(chunk) + 1)) if isinstance(raw, dict) else 0} "
                  f"items via {gem.version} ({gem.calls} calls)", flush=True)
    except QuotaExhausted as e:
        print(f"Stopping: {e}. Re-run tomorrow; finished passages are cached.", flush=True)


def export() -> dict:
    records, stats = [], Counter()
    for path in sorted(CACHE.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        p, meta = data["passage"], BOOKS[data["passage"]["book"]]
        if not data["items"]:
            stats["passage_skipped_by_writer"] += 1
        for j, item in enumerate(data["items"]):
            if not valid(item):
                stats["invalid"] += 1
                continue
            if item["type"] in ("mcq", "tf"):
                key = str(item["key"]).upper() if item["type"] == "mcq" else ("TRUE" if item["key"] else "FALSE")
                if stated_answer(item["answer"]) != key:
                    stats["key_mismatch"] += 1
                    continue
            stats["kept"] += 1
            records.append({
                "id": f"textbook-{p['id']}-{j}", "subject": p["subject"], "topic": f"textbook/{p['book']}/{p['section']}",
                "source": "textbook", "format": item["type"], "teacher": data["model"],
                "attribution": f"{meta['authors']}, {meta['title']} ({meta['license']}), {meta['url']}",
                "passage_sha1": hashlib.sha1(p["text"].encode()).hexdigest()[:12],
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user_prompt(item)},
                             {"role": "assistant", "content": item["answer"].strip()}]})
    write_jsonl(records, OUT)
    stats["by_book"] = dict(Counter(r["topic"].split("/")[1] for r in records))
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-book", type=int, default=60, help="passages sampled per book")
    ap.add_argument("--per-call", type=int, default=6, help="passages per Gemini request")
    ap.add_argument("--n-open", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--rpm", type=float, default=8)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    if args.export_only:
        print("export:", export())
        return
    selected = []
    for book in BOOKS:
        ps = passages(book)
        pick = select(ps, args.per_book)
        selected += pick
        print(f"{book}: {len(ps)} passages, {len(pick)} selected "
              f"({sum(len(p['text'].split()) for p in pick):,} words)", flush=True)
    (BOOKS_DIR / "selected_passages.json").write_text(json.dumps(selected, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
    if args.prepare_only:
        return
    generate(selected, args.models, args.rpm, args.per_call, args.n_open)
    print("export:", export())


if __name__ == "__main__":
    main()
