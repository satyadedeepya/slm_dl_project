"""Progress of the final stage (FT-C deep-learning training + evaluation), read from outputs/ logs.

  python train/ftc_progress.py --short   # one line (status line)
  python train/ftc_progress.py --watch   # refresh every 30 s
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
EVAL_TASKS = 8  # gen_dl, gen_dbms, gen_cn, spider, mbpp, csbench, gen_os, gate
STAGES = [("train", "FT-C training", 1.2), ("eval", "FT-C evaluation", 2.5)]


def _read(name: str) -> str:
    try:
        return (OUT / name).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def train_progress() -> tuple[float, str]:
    if (OUT / "ft-c" / "adapter").exists():
        return 1.0, "done"
    bars = re.findall(r"(\d+)/(\d+) \[[^\]]*?<[0-9:]+, *([0-9.]+)s/it", _read("ft-c.log").replace("\r", "\n"))
    bars = [b for b in bars if int(b[1]) != 100]  # skip the validation bars
    if not bars:
        return 0.0, "starting"
    done, total = int(bars[-1][0]), int(bars[-1][1])
    rate = sorted(float(b[2]) for b in bars[-10:])[len(bars[-10:]) // 2]
    left = int((total - done) * rate)
    return done / total, f"step {done}/{total}, ~{left // 3600}h{left % 3600 // 60:02d}m left"


def eval_progress() -> tuple[float, str]:
    n = len(re.findall(r"^\s+\w+: acc=", _read("eval-ft-c.log"), re.M))
    return min(n, EVAL_TASKS) / EVAL_TASKS, f"{n}/{EVAL_TASKS} tasks"


def status():
    values = {"train": train_progress(), "eval": eval_progress()}
    total = sum(h for *_, h in STAGES)
    overall = sum(values[k][0] * h for k, _, h in STAGES) / total
    return overall, [(label, *values[k]) for k, label, _ in STAGES]


def short() -> str:
    overall, rows = status()
    label, _, info = next((r for r in rows if r[1] < 1), ("FT-C", 1, "complete"))
    bar = "█" * round(overall * 10) + "░" * (10 - round(overall * 10))
    return f"DL stage {overall * 100:.0f}% {bar} | {label}: {info}"


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if "--watch" in sys.argv:
        while True:
            print("\033[2J\033[H" + time.strftime("%H:%M:%S  ") + short(), flush=True)
            time.sleep(30)
    else:
        print(short())
