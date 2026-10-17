"""Verified Operating Systems problem generator.

Problem families (answers are computed by simulation, never by an LLM):
  * CPU scheduling: FCFS, SJF, SRTF, Round Robin, Priority (non-/preemptive)
  * Page replacement: FIFO, LRU, Optimal
  * Deadlock avoidance: Banker's safety algorithm
  * Disk scheduling: FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK

Usage:
  python -m data.generators.os_gen --n 900 --seed 42 --split train --out data/processed/gen_os_train.jsonl
"""
from __future__ import annotations

import random
from fractions import Fraction

from .common import Problem, cli, fmt, fmt_avg, md_table, run_generators, write_jsonl  # noqa: F401

# =====================================================================
# CPU scheduling
# =====================================================================

ALGO_NAMES = {
    "fcfs": "First-Come First-Served (FCFS)",
    "sjf": "Shortest Job First (SJF, non-preemptive)",
    "srtf": "Shortest Remaining Time First (SRTF, preemptive SJF)",
    "rr": "Round Robin (RR)",
    "prio": "Priority scheduling (non-preemptive)",
    "prio_p": "Priority scheduling (preemptive)",
}

ALGO_RULES = {
    "fcfs": "Run processes in order of arrival time (ties: lower process number first); each runs to completion.",
    "sjf": "Whenever the CPU becomes free, pick the arrived process with the smallest burst time "
           "(ties: earlier arrival, then lower process number). It runs to completion.",
    "srtf": "At every time unit, run the arrived process with the smallest remaining time "
            "(ties: earlier arrival, then lower process number). A new arrival can preempt the running process.",
    "rr": "Processes wait in a FIFO ready queue and run for at most one time quantum. Processes that arrive "
          "during a slice join the queue before the preempted process is put back at the tail.",
    "prio": "Whenever the CPU becomes free, pick the arrived process with the highest priority "
            "(smallest priority number; ties: earlier arrival, then lower process number). It runs to completion.",
    "prio_p": "At every time unit, run the arrived process with the highest priority (smallest priority number; "
              "ties: earlier arrival, then lower process number). A higher-priority arrival preempts the running process.",
}


def _merge(gantt: list[list]) -> list[tuple]:
    out: list[list] = []
    for name, s, e in gantt:
        if out and out[-1][0] == name and out[-1][2] == s:
            out[-1][2] = e
        else:
            out.append([name, s, e])
    return [tuple(g) for g in out]


def schedule(procs: list[dict], algo: str, quantum: int | None = None):
    """Simulate a scheduler.

    procs: [{"name": "P1", "at": 0, "bt": 5, "pr": 2}, ...] (pr optional)
    Returns (gantt, ct, first) where gantt = [(name|None for idle, start, end)],
    ct = {name: completion time}, first = {name: first time on CPU}.
    """
    idx = {p["name"]: i for i, p in enumerate(procs)}
    ct: dict[str, int] = {}
    first: dict[str, int] = {}
    gantt: list[list] = []

    if algo in ("fcfs", "sjf", "prio"):
        key = {
            "fcfs": lambda p: (p["at"], idx[p["name"]]),
            "sjf": lambda p: (p["bt"], p["at"], idx[p["name"]]),
            "prio": lambda p: (p["pr"], p["at"], idx[p["name"]]),
        }[algo]
        remaining = list(procs)
        t = 0
        while remaining:
            ready = [p for p in remaining if p["at"] <= t]
            if not ready:
                nxt = min(p["at"] for p in remaining)
                gantt.append([None, t, nxt])
                t = nxt
                continue
            p = min(ready, key=key)
            first[p["name"]] = t
            gantt.append([p["name"], t, t + p["bt"]])
            t += p["bt"]
            ct[p["name"]] = t
            remaining.remove(p)

    elif algo in ("srtf", "prio_p"):
        rem = {p["name"]: p["bt"] for p in procs}
        key = {
            "srtf": lambda p: (rem[p["name"]], p["at"], idx[p["name"]]),
            "prio_p": lambda p: (p["pr"], p["at"], idx[p["name"]]),
        }[algo]
        t = 0
        while len(ct) < len(procs):
            ready = [p for p in procs if p["at"] <= t and rem[p["name"]] > 0]
            if not ready:
                gantt.append([None, t, t + 1])
                t += 1
                continue
            p = min(ready, key=key)
            first.setdefault(p["name"], t)
            gantt.append([p["name"], t, t + 1])
            rem[p["name"]] -= 1
            t += 1
            if rem[p["name"]] == 0:
                ct[p["name"]] = t

    elif algo == "rr":
        assert quantum and quantum > 0
        order = sorted(procs, key=lambda p: (p["at"], idx[p["name"]]))
        rem = {p["name"]: p["bt"] for p in procs}
        queue: list[dict] = []
        arrived = 0
        t = 0

        def admit(upto: int):
            nonlocal arrived
            while arrived < len(order) and order[arrived]["at"] <= upto:
                queue.append(order[arrived])
                arrived += 1

        admit(t)
        while len(ct) < len(procs):
            if not queue:
                nxt = order[arrived]["at"]
                gantt.append([None, t, nxt])
                t = nxt
                admit(t)
                continue
            p = queue.pop(0)
            run = min(quantum, rem[p["name"]])
            first.setdefault(p["name"], t)
            gantt.append([p["name"], t, t + run])
            t += run
            rem[p["name"]] -= run
            admit(t)  # arrivals during/at end of the slice go before the preempted process
            if rem[p["name"]] > 0:
                queue.append(p)
            else:
                ct[p["name"]] = t
    else:
        raise ValueError(algo)

    return _merge(gantt), ct, first


def scheduling_metrics(procs, ct, first):
    rows = []
    for p in procs:
        tat = ct[p["name"]] - p["at"]
        rows.append({**p, "ct": ct[p["name"]], "tat": tat, "wt": tat - p["bt"],
                     "rt": first[p["name"]] - p["at"]})
    n = len(rows)
    return rows, sum(r["wt"] for r in rows), sum(r["tat"] for r in rows), n


def _gantt_str(gantt) -> str:
    return " → ".join(f"{name or 'idle'} [{s}–{e}]" for name, s, e in gantt)


Q_SCHED = [
    "Consider the following processes with their arrival times and CPU burst times{pr_note}:\n\n{table}\n\n"
    "Using {algo}{q_note}, draw the Gantt chart and compute the average waiting time and average turnaround time.",
    "{n} processes arrive at a single-CPU system as shown below{pr_note}.\n\n{table}\n\n"
    "Schedule them with {algo}{q_note}. What are the average waiting time and the average turnaround time?",
    "Given the process table below{pr_note}, apply {algo}{q_note}. Show the Gantt chart, the completion, "
    "turnaround and waiting time of every process, and the averages.\n\n{table}",
]


def gen_scheduling(rng: random.Random, split: str = "train") -> Problem:
    algo = rng.choice(list(ALGO_NAMES))
    n = rng.randint(3, 5) if split == "train" else rng.randint(4, 6)
    max_bt = 10 if split == "train" else 12
    procs = []
    for i in range(n):
        p = {"name": f"P{i + 1}", "at": 0 if i == 0 else rng.randint(0, 8), "bt": rng.randint(1, max_bt)}
        if algo.startswith("prio"):
            p["pr"] = rng.randint(1, 5)
        procs.append(p)
    quantum = rng.randint(2, 4) if algo == "rr" else None

    gantt, ct, first = schedule(procs, algo, quantum)
    rows, wt_sum, tat_sum, n = scheduling_metrics(procs, ct, first)

    has_pr = algo.startswith("prio")
    in_header = ["Process", "Arrival time", "Burst time"] + (["Priority"] if has_pr else [])
    in_rows = [[p["name"], p["at"], p["bt"]] + ([p["pr"]] if has_pr else []) for p in procs]
    question = rng.choice(Q_SCHED).format(
        n=n,
        table=md_table(in_header, in_rows),
        algo=ALGO_NAMES[algo],
        q_note=f" with time quantum = {quantum}" if quantum else "",
        pr_note=" (a smaller priority number means higher priority)" if has_pr else "",
    )

    out_rows = [[r["name"], r["at"], r["bt"], r["ct"], f"{r['ct']} − {r['at']} = {r['tat']}",
                 f"{r['tat']} − {r['bt']} = {r['wt']}", r["rt"]] for r in rows]
    avg_wt, avg_tat = Fraction(wt_sum, n), Fraction(tat_sum, n)
    solution = (
        f"**Rule ({ALGO_NAMES[algo]}{', q = ' + str(quantum) if quantum else ''}):** {ALGO_RULES[algo]}\n\n"
        f"**Gantt chart:** {_gantt_str(gantt)}\n\n"
        "Turnaround time (TAT) = Completion time (CT) − Arrival time (AT); "
        "Waiting time (WT) = TAT − Burst time (BT); Response time (RT) = first CPU time − AT.\n\n"
        + md_table(["Process", "AT", "BT", "CT", "TAT", "WT", "RT"], out_rows)
        + f"\n\nAverage waiting time = {fmt_avg(wt_sum, n)}\n"
        f"Average turnaround time = {fmt_avg(tat_sum, n)}\n\n"
        f"**Final answer:** Average waiting time = {fmt(avg_wt)}, Average turnaround time = {fmt(avg_tat)}"
    )
    return Problem("OS", f"cpu_scheduling/{algo}", question, solution,
                   {"type": "numbers", "values": [float(avg_wt), float(avg_tat)]})


# =====================================================================
# Page replacement
# =====================================================================

def page_replacement(refs: list[int], frames: int, algo: str):
    """Returns (faults, steps) where steps = [(ref, frame_slots_after, hit)]."""
    slots: list[int | None] = [None] * frames
    loaded_at: dict[int, int] = {}   # FIFO: time the page was loaded
    last_used: dict[int, int] = {}   # LRU
    faults, steps = 0, []
    for t, page in enumerate(refs):
        hit = page in slots
        if not hit:
            faults += 1
            if None in slots:
                victim_slot = slots.index(None)
            elif algo == "fifo":
                victim_slot = min(range(frames), key=lambda s: loaded_at[slots[s]])
            elif algo == "lru":
                victim_slot = min(range(frames), key=lambda s: last_used[slots[s]])
            elif algo == "optimal":
                def next_use(s):
                    try:
                        return refs.index(slots[s], t + 1)
                    except ValueError:
                        return float("inf")
                victim_slot = max(range(frames), key=lambda s: (next_use(s), -s))
            else:
                raise ValueError(algo)
            slots[victim_slot] = page
            loaded_at[page] = t
        last_used[page] = t
        steps.append((page, list(slots), hit))
    return faults, steps


PR_NAMES = {"fifo": "FIFO", "lru": "LRU (Least Recently Used)", "optimal": "Optimal (OPT / Belady's)"}
PR_RULES = {
    "fifo": "On a fault with all frames full, replace the page that was loaded earliest.",
    "lru": "On a fault with all frames full, replace the page whose last use is furthest in the past.",
    "optimal": "On a fault with all frames full, replace the page whose next use is furthest in the future "
               "(or that is never used again).",
}
Q_PR = [
    "Consider the page reference string: {refs}\nThe system has {f} page frames, initially empty. "
    "How many page faults occur with the {algo} page replacement algorithm?",
    "A process makes the following page references: {refs}. With {f} frames (all empty at the start) and "
    "{algo} replacement, find the number of page faults and the page fault rate.",
    "Using {algo} page replacement with {f} initially empty frames, trace the reference string "
    "{refs} and count the page faults.",
]


def gen_page_replacement(rng: random.Random, split: str = "train") -> Problem:
    algo = rng.choice(list(PR_NAMES))
    length = rng.randint(10, 16) if split == "train" else rng.randint(14, 20)
    refs = [rng.randint(0, 7) for _ in range(length)]
    f = rng.choice([3, 4]) if split == "train" else rng.choice([3, 4, 5])
    faults, steps = page_replacement(refs, f, algo)

    rows = [[i + 1, ref] + [("–" if s is None else s) for s in slots] + ["Hit" if hit else "**Fault**"]
            for i, (ref, slots, hit) in enumerate(steps)]
    header = ["Step", "Ref"] + [f"F{i + 1}" for i in range(f)] + ["Result"]
    solution = (
        f"**Rule ({PR_NAMES[algo]}):** {PR_RULES[algo]}\n\n"
        + md_table(header, rows)
        + f"\n\nPage faults = {faults}, hits = {len(refs) - faults} out of {len(refs)} references.\n"
        f"Page fault rate = {faults}/{len(refs)} = {faults / len(refs):.2f}\n\n"
        f"**Final answer:** {faults} page faults"
    )
    question = rng.choice(Q_PR).format(refs=", ".join(map(str, refs)), f=f, algo=PR_NAMES[algo])
    return Problem("OS", f"page_replacement/{algo}", question, solution,
                   {"type": "numbers", "values": [faults]})


# =====================================================================
# Banker's algorithm
# =====================================================================

def bankers_safety(alloc, maxm, avail):
    """Classic pass-based safety algorithm.

    Returns (safe, sequence, trace) where trace = [(process_index, work_after)].
    """
    n, m = len(alloc), len(avail)
    need = [[maxm[i][j] - alloc[i][j] for j in range(m)] for i in range(n)]
    work, finish, seq, trace = list(avail), [False] * n, [], []
    progress = True
    while progress:
        progress = False
        for i in range(n):
            if not finish[i] and all(need[i][j] <= work[j] for j in range(m)):
                work = [work[j] + alloc[i][j] for j in range(m)]
                finish[i] = True
                seq.append(i)
                trace.append((i, list(work)))
                progress = True
    return all(finish), seq, trace


def is_safe_sequence(alloc, maxm, avail, seq: list[int]) -> bool:
    """Verifier: is `seq` (0-based process indices) a valid safe sequence?"""
    n, m = len(alloc), len(avail)
    if sorted(seq) != list(range(n)):
        return False
    work = list(avail)
    for i in seq:
        if any(maxm[i][j] - alloc[i][j] > work[j] for j in range(m)):
            return False
        work = [work[j] + alloc[i][j] for j in range(m)]
    return True


Q_BANK = [
    "A system has {m} resource types ({res}) and {n} processes. The current state is:\n\n{table}\n\n"
    "Available = ({avail}). Using the Banker's algorithm, determine whether the system is in a safe state. "
    "If it is, give a safe sequence.",
    "Apply the Banker's safety algorithm to the following snapshot. Is the state safe? If yes, list a safe "
    "sequence; if not, explain why.\n\n{table}\n\nAvailable resources ({res}) = ({avail})",
]


def gen_bankers(rng: random.Random, split: str = "train") -> Problem:
    n = rng.randint(4, 5) if split == "train" else rng.randint(5, 6)
    m = 3
    want_safe = rng.random() < 0.7
    while True:
        maxm = [[rng.randint(0, 9) for _ in range(m)] for _ in range(n)]
        alloc = [[rng.randint(0, x) for x in row] for row in maxm]
        avail = [rng.randint(0, 5) for _ in range(m)]
        safe, seq, trace = bankers_safety(alloc, maxm, avail)
        # Unsafe cases must get partway (>= 2 processes finish) so the trace is instructive.
        if safe == want_safe and (safe or len(seq) >= 2):
            break
    res = [chr(ord("A") + j) for j in range(m)]
    need = [[maxm[i][j] - alloc[i][j] for j in range(m)] for i in range(n)]
    vec = lambda v: " ".join(map(str, v))
    table = md_table(["Process", f"Allocation ({' '.join(res)})", f"Max ({' '.join(res)})"],
                     [[f"P{i}", vec(alloc[i]), vec(maxm[i])] for i in range(n)])
    question = rng.choice(Q_BANK).format(m=m, n=n, res=", ".join(res), table=table, avail=", ".join(map(str, avail)))

    lines = [
        "**Step 1: Need = Max − Allocation**\n",
        md_table(["Process", f"Need ({' '.join(res)})"], [[f"P{i}", vec(need[i])] for i in range(n)]),
        f"\n**Step 2: Safety algorithm.** Work = Available = ({vec(avail)}). Repeatedly scan P0..P{n - 1} "
        "and run any unfinished process whose Need ≤ Work, then add its Allocation to Work.\n",
    ]
    for i, work in trace:
        lines.append(f"- P{i}: Need ({vec(need[i])}) ≤ Work → finishes, Work = ({vec(work)})")
    if safe:
        seq_s = ", ".join(f"P{i}" for i in seq)
        lines.append(f"\nAll processes can finish, so the system is in a **safe state**.\n\n"
                     f"**Final answer:** Safe; safe sequence = <{seq_s}>")
    else:
        stuck = [f"P{i}" for i in range(n) if i not in seq]
        lines.append(f"\nNo remaining process ({', '.join(stuck)}) has Need ≤ Work = ({vec(trace[-1][1] if trace else avail)}), "
                     "so they can never finish.\n\n**Final answer:** Unsafe; no safe sequence exists")
    return Problem("OS", "deadlock/bankers", question, "\n".join(lines),
                   {"type": "bankers", "safe": safe, "alloc": alloc, "max": maxm, "avail": avail})


# =====================================================================
# Disk scheduling
# =====================================================================

DISK_NAMES = {"fcfs": "FCFS", "sstf": "SSTF", "scan": "SCAN", "cscan": "C-SCAN", "look": "LOOK", "clook": "C-LOOK"}
DISK_RULES = {
    "fcfs": "Serve requests in the order they arrive.",
    "sstf": "Always serve the pending request closest to the current head position (ties: lower cylinder).",
    "scan": "Move in the given direction serving requests, continue to the end of the disk, then reverse.",
    "cscan": "Move in the given direction serving requests up to the end of the disk, jump to the other end "
             "(the return seek is counted), and continue in the same direction.",
    "look": "Move in the given direction only as far as the last request, then reverse.",
    "clook": "Move in the given direction up to the last request, jump to the furthest request at the other "
             "end (the jump is counted), and continue in the same direction.",
}


def disk_schedule(head: int, reqs: list[int], algo: str, direction: str = "up", max_cyl: int = 199) -> list[int]:
    """Returns the path of head positions (starting with `head`), including end-of-disk visits."""
    up = sorted(r for r in reqs if r >= head)
    down = sorted((r for r in reqs if r < head), reverse=True)
    if direction == "down":
        up, down = sorted((r for r in reqs if r <= head), reverse=True), sorted(r for r in reqs if r > head)
    end, other_end = (max_cyl, 0) if direction == "up" else (0, max_cyl)

    if algo == "fcfs":
        path = list(reqs)
    elif algo == "sstf":
        pending, pos, path = list(reqs), head, []
        while pending:
            nxt = min(pending, key=lambda r: (abs(r - pos), r))
            pending.remove(nxt)
            path.append(nxt)
            pos = nxt
    elif algo == "scan":
        path = up + ([end] if down and (not up or up[-1] != end) else []) + down
    elif algo == "look":
        path = up + down
    elif algo == "cscan":
        rest = sorted(down) if direction == "up" else sorted(down, reverse=True)
        path = up + ([end] if not up or up[-1] != end else []) + [other_end] + rest
    elif algo == "clook":
        rest = sorted(down) if direction == "up" else sorted(down, reverse=True)
        path = up + rest
    else:
        raise ValueError(algo)
    return [head] + path


def head_movement(path: list[int]) -> int:
    return sum(abs(b - a) for a, b in zip(path, path[1:]))


Q_DISK = [
    "A disk has cylinders numbered 0 to {mc}. The disk head is at cylinder {head}{dir_note}. The request queue "
    "(in arrival order) is: {reqs}. Using {algo} disk scheduling, find the total head movement in cylinders.",
    "Disk requests arrive for cylinders {reqs} (in this order). The head starts at {head}{dir_note}, and the "
    "disk has cylinders 0–{mc}. Compute the total number of cylinders traversed by the head under {algo}.",
]


def gen_disk(rng: random.Random, split: str = "train") -> Problem:
    algo = rng.choice(list(DISK_NAMES))
    mc = 199
    k = rng.randint(6, 8) if split == "train" else rng.randint(8, 10)
    head = rng.randint(30, 170)
    while True:  # require requests on both sides of the head so direction conventions matter
        reqs = rng.sample([c for c in range(mc + 1) if c != head], k)
        if any(r < head for r in reqs) and any(r > head for r in reqs):
            break
    direction = rng.choice(["up", "down"])
    path = disk_schedule(head, reqs, algo, direction, mc)
    total = head_movement(path)

    dir_word = "towards higher cylinder numbers" if direction == "up" else "towards lower cylinder numbers"
    dir_note = f" and moving {dir_word}" if algo not in ("fcfs", "sstf") else ""
    question = rng.choice(Q_DISK).format(mc=mc, head=head, reqs=", ".join(map(str, reqs)),
                                         algo=DISK_NAMES[algo], dir_note=dir_note)
    if algo in ("cscan", "clook"):
        question += " Count the return seek as head movement."
    moves = " + ".join(f"|{b} − {a}|" for a, b in zip(path, path[1:]))
    vals = " + ".join(str(abs(b - a)) for a, b in zip(path, path[1:]))
    solution = (
        f"**Rule ({DISK_NAMES[algo]}{', moving ' + dir_word if dir_note else ''}):** {DISK_RULES[algo]}\n\n"
        f"**Head path:** {' → '.join(map(str, path))}\n\n"
        f"Total head movement = {moves}\n= {vals}\n= {total} cylinders\n\n"
        f"**Final answer:** {total} cylinders"
    )
    return Problem("OS", f"disk_scheduling/{algo}", question, solution,
                   {"type": "numbers", "values": [total]})


# =====================================================================
# Driver
# =====================================================================

GENERATORS = [(gen_scheduling, 0.40), (gen_page_replacement, 0.25), (gen_bankers, 0.15), (gen_disk, 0.20)]


def generate(n: int, seed: int, split: str = "train") -> list[dict]:
    return run_generators(GENERATORS, "os", n, seed, split)


if __name__ == "__main__":
    cli(GENERATORS, "os", "OS", default_n=900)
