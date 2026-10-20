"""Week 3 progress meter, read from the training/evaluation logs in outputs/.

  python train/week3_progress.py           # multi-line view (live window)
  python train/week3_progress.py --short   # one line (Claude Code status line)
  python train/week3_progress.py --watch   # refresh every 30 s

Overall % weights each stage by its expected duration (hours).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
EVAL_TASKS = 7  # csbench, gen_os, gen_cn, gen_dbms, mbpp, gate, spider
STAGES = [  # (key, label, expected hours)
    ("setup", "Setup (script, timing, runner)", 1.0),
    ("train_b", "FT-B training", 6.0),
    ("eval_b", "FT-B evaluation", 3.0),
    ("train_a", "FT-A training", 4.0),
    ("eval_a", "FT-A evaluation", 3.0),
    ("report", "Analysis + report + push", 1.0),
]


def _read(name: str) -> str:
    p = OUT / name
    try:
        return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    except OSError:
        return ""


def train_progress(run: str) -> tuple[float, str]:
    if (OUT / run / "adapter").exists():
        return 1.0, "done"
    bars = re.findall(r"(\d+)/(\d+) \[[^\]]*?<[0-9:]+, *([0-9.]+)s/it", _read(f"{run}.log").replace("\r", "\n"))
    bars = [b for b in bars if int(b[1]) > 150]  # skip the 150-example validation bars
    if not bars:
        return 0.0, "waiting"
    done, total = int(bars[-1][0]), int(bars[-1][1])
    # tqdm's s/it is a running average that a sleep/hibernate pause inflates; use the median recent rate
    rate = sorted(float(b[2]) for b in bars[-10:])[len(bars[-10:]) // 2]
    left = int((total - done) * rate)
    return done / total, f"step {done}/{total}, ~{left // 3600}h{left % 3600 // 60:02d}m left"


def eval_progress(log: str) -> tuple[float, str]:
    text = _read(log)
    if not text:
        return 0.0, "waiting"
    n = len(re.findall(r"^\s+\w+: acc=", text, re.M))
    return n / EVAL_TASKS, f"{n}/{EVAL_TASKS} tasks"


def status() -> tuple[float, list[tuple[str, float, str]]]:
    rows = []
    tb, ib = train_progress("ft-b")
    eb, jb = eval_progress("eval-ft-b.log")
    ta, ia = train_progress("ft-a")
    ea, ja = eval_progress("eval-ft-a.log")
    report = 1.0 if (OUT / "week3.report.done").exists() else (0.6 if (OUT / "week3.done").exists() else 0.3)
    values = {"setup": (1.0, "done"), "train_b": (tb, ib), "eval_b": (eb, jb), "train_a": (ta, ia),
              "eval_a": (ea, ja), "report": (report, "docs/week3_report.md")}
    total = sum(h for _, _, h in STAGES)
    overall = sum(values[k][0] * h for k, _, h in STAGES) / total
    for k, label, _ in STAGES:
        rows.append((label, *values[k]))
    return overall, rows


def bar(frac: float, width: int = 20) -> str:
    n = round(frac * width)
    return "█" * n + "░" * (width - n)


def short() -> str:
    overall, rows = status()
    current = next(((label, info) for label, f, info in rows if f < 1), ("Week 3", "complete"))
    return f"Week 3 {overall * 100:.0f}% {bar(overall, 10)} | left {100 - overall * 100:.0f}% | {current[0]}: {current[1]}"


def full() -> str:
    overall, rows = status()
    lines = [f"WEEK 3 PROGRESS  {time.strftime('%d %b %H:%M:%S')}", "",
             f"  Overall  [{bar(overall)}] {overall * 100:5.1f}%   remaining {100 - overall * 100:5.1f}%", ""]
    for label, f, info in rows:
        lines.append(f"  {label:<32} [{bar(f, 14)}] {f * 100:5.1f}%  {info}")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if "--short" in sys.argv:
        print(short())
    elif "--watch" in sys.argv:
        while True:
            print("\033[2J\033[H" + full(), flush=True)
            time.sleep(30)
    else:
        print(full())
