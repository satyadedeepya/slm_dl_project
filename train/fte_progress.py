"""Progress of FT-E (fuller explanation stage), from answer expansion to the final judge, read from outputs/ logs.

  python train/fte_progress.py --short   # one line (status line)
  python train/fte_progress.py --watch   # refresh every 30 s
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EXPANDED = ROOT / "data" / "concepts" / "explain_cache_v2"
EXPLAIN = ROOT / "eval" / "results" / "explain"
EVAL_TASKS = 5  # gen_dl, gen_dbms, gen_cn, spider, mbpp
JUDGE_CALLS = 78  # 39 questions x 2 orders
# (key, label, expected hours): the weights for the overall percentage
STAGES = [("expand", "expanding answers", 0.75), ("train", "FT-E training", 1.9), ("eval", "accuracy tests", 0.6),
          ("answer", "explanation answers", 0.15), ("judge", "blind judge", 0.4)]
DEADLINE = "23:30"  # expansion is time-boxed (train/run_fte.cmd)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _hm(seconds: float) -> str:
    s = int(seconds)
    return f"~{s // 3600}h{s % 3600 // 60:02d}m left"


def _short_batches(min_words: int = 220, batch: int = 5) -> int:
    """Batches the expansion needs in total: short teacher answers (as in build_explain_stage.py expand)."""
    import json
    n = 0
    for p in (ROOT / "data" / "concepts" / "explain_cache").glob("*.json"):
        n += sum(1 for a in json.loads(p.read_text(encoding="utf-8"))["answers"].values()
                 if a and len(a.split()) < min_words)
    return max(1, -(-n // batch))


def expand_progress() -> tuple[float, str]:
    if (OUT / "ft-e.log").exists():
        return 1.0, "done"
    if not re.search(r"in (\d+) batches", _read(OUT / "explain_expand.log")):
        return 0.0, "starting"
    total = _short_batches()  # the log only counts what is left after a restart, so count from the data
    times = sorted(p.stat().st_mtime for p in EXPANDED.glob("part_*.json"))
    done = min(len(times), total)
    now = time.localtime()
    h, m = map(int, DEADLINE.split(":"))
    mins_left = h * 60 + m - (now.tm_hour * 60 + now.tm_min)
    if mins_left <= 45:  # time-boxed: progress follows the clock in the last 45 minutes
        return max(done / total, 1 - max(mins_left, 0) / 45), f"{done}/{total} batches, stops at {DEADLINE}"
    recent = times[-20:]
    rate = (recent[-1] - recent[0]) / (len(recent) - 1) if len(recent) > 1 else 0  # seconds per batch
    left = f", {_hm((total - done) * rate)}" if rate else ""
    return done / total, f"{done}/{total} batches{left}"


def train_progress() -> tuple[float, str]:
    if (OUT / "ft-e" / "adapter").exists():
        return 1.0, "done"
    bars = re.findall(r"(\d+)/(\d+) \[[^\]]*?<[0-9:]+, *([0-9.]+)s/it", _read(OUT / "ft-e.log").replace("\r", "\n"))
    bars = [b for b in bars if int(b[1]) != 100]  # skip the validation bars
    if not bars:
        return 0.0, "starting" if (OUT / "ft-e.log").exists() else "waiting"
    done, total = int(bars[-1][0]), int(bars[-1][1])
    rate = sorted(float(b[2]) for b in bars[-10:])[len(bars[-10:]) // 2]
    return done / total, f"step {done}/{total}, {_hm((total - done) * rate)}"


def eval_progress() -> tuple[float, str]:
    n = len(re.findall(r"^\s+\w+: acc=", _read(OUT / "eval-ft-e.log"), re.M))
    return min(n, EVAL_TASKS) / EVAL_TASKS, f"{n}/{EVAL_TASKS} tasks"


def answer_progress() -> tuple[float, str]:
    if (EXPLAIN / "answers_ft-e.jsonl").exists():
        return 1.0, "done"
    m = re.findall(r"generated (\d+)/(\d+)", _read(OUT / "explain-ft-e.log"))
    return (int(m[-1][0]) / int(m[-1][1]), f"{m[-1][0]}/{m[-1][1]}") if m else (0.0, "waiting")


def judge_progress() -> tuple[float, str]:
    n = len(_read(EXPLAIN / "judge_base_vs_ft-e.jsonl").splitlines())
    if (OUT / "fte.done").exists():
        return 1.0, "done"
    return min(n, JUDGE_CALLS) / JUDGE_CALLS, f"{n}/{JUDGE_CALLS} comparisons"


def status():
    values = {"expand": expand_progress(), "train": train_progress(), "eval": eval_progress(),
              "answer": answer_progress(), "judge": judge_progress()}
    total = sum(h for *_, h in STAGES)
    overall = sum(values[k][0] * h for k, _, h in STAGES) / total
    return overall, [(label, *values[k]) for k, label, _ in STAGES]


def short() -> str:
    overall, rows = status()
    label, _, info = next((r for r in rows if r[1] < 1), ("FT-E", 1, "complete"))
    bar = "█" * round(overall * 10) + "░" * (10 - round(overall * 10))
    return f"FT-E {overall * 100:.0f}% {bar} | {label}: {info}"


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if "--watch" in sys.argv:
        while True:
            print("\033[2J\033[H" + time.strftime("%H:%M:%S  ") + short(), flush=True)
            time.sleep(30)
    else:
        print(short())
