"""Checks the OS solvers against worked examples from Silberschatz (Galvin), plus
property tests over thousands of generated problems.

Run: python -m unittest discover -s tests -v
"""
import random
import unittest
from fractions import Fraction

from data.generators import os_gen
from data.generators.grading import grade
from data.generators.os_gen import (bankers_safety, disk_schedule, head_movement, is_safe_sequence,
                                    page_replacement, schedule, scheduling_metrics)


def avg_wt(procs, algo, q=None):
    gantt, ct, first = schedule(procs, algo, q)
    _, wt, tat, n = scheduling_metrics(procs, ct, first)
    return Fraction(wt, n), Fraction(tat, n)


def P(name, at, bt, pr=None):
    d = {"name": name, "at": at, "bt": bt}
    if pr is not None:
        d["pr"] = pr
    return d


class TestScheduling(unittest.TestCase):
    def test_fcfs_with_arrivals(self):
        wt, tat = avg_wt([P("P1", 0, 5), P("P2", 1, 3), P("P3", 2, 8)], "fcfs")
        self.assertEqual((wt, tat), (Fraction(10, 3), Fraction(26, 3)))

    def test_fcfs_galvin(self):  # 24, 3, 3 -> avg WT 17
        wt, _ = avg_wt([P("P1", 0, 24), P("P2", 0, 3), P("P3", 0, 3)], "fcfs")
        self.assertEqual(wt, 17)

    def test_sjf_galvin(self):  # 6, 8, 7, 3 -> avg WT 7
        wt, _ = avg_wt([P("P1", 0, 6), P("P2", 0, 8), P("P3", 0, 7), P("P4", 0, 3)], "sjf")
        self.assertEqual(wt, 7)

    def test_srtf_galvin(self):  # avg WT 26/4 = 6.5
        procs = [P("P1", 0, 8), P("P2", 1, 4), P("P3", 2, 9), P("P4", 3, 5)]
        wt, _ = avg_wt(procs, "srtf")
        self.assertEqual(wt, Fraction(13, 2))
        gantt, _, _ = schedule(procs, "srtf")
        self.assertEqual(gantt, [("P1", 0, 1), ("P2", 1, 5), ("P4", 5, 10), ("P1", 10, 17), ("P3", 17, 26)])

    def test_rr_galvin(self):  # q=4 -> avg WT 17/3
        wt, _ = avg_wt([P("P1", 0, 24), P("P2", 0, 3), P("P3", 0, 3)], "rr", 4)
        self.assertEqual(wt, Fraction(17, 3))

    def test_rr_arrival_goes_before_preempted(self):
        # P1 runs 0-2; P2 arrives at 1 so it is queued before P1 returns.
        gantt, _, _ = schedule([P("P1", 0, 4), P("P2", 1, 2)], "rr", 2)
        self.assertEqual(gantt, [("P1", 0, 2), ("P2", 2, 4), ("P1", 4, 6)])

    def test_priority_galvin(self):  # avg WT 8.2
        procs = [P("P1", 0, 10, 3), P("P2", 0, 1, 1), P("P3", 0, 2, 4), P("P4", 0, 1, 5), P("P5", 0, 5, 2)]
        wt, _ = avg_wt(procs, "prio")
        self.assertEqual(wt, Fraction(41, 5))

    def test_idle_gap(self):
        gantt, ct, _ = schedule([P("P1", 0, 2), P("P2", 5, 3)], "fcfs")
        self.assertEqual(gantt, [("P1", 0, 2), (None, 2, 5), ("P2", 5, 8)])
        self.assertEqual(ct["P2"], 8)

    def test_preemptive_priority(self):
        gantt, _, _ = schedule([P("P1", 0, 5, 3), P("P2", 2, 2, 1)], "prio_p")
        self.assertEqual(gantt, [("P1", 0, 2), ("P2", 2, 4), ("P1", 4, 7)])


GALVIN_REFS = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


class TestPageReplacement(unittest.TestCase):
    def test_galvin_3_frames(self):
        self.assertEqual(page_replacement(GALVIN_REFS, 3, "fifo")[0], 15)
        self.assertEqual(page_replacement(GALVIN_REFS, 3, "lru")[0], 12)
        self.assertEqual(page_replacement(GALVIN_REFS, 3, "optimal")[0], 9)

    def test_belady_anomaly(self):
        refs = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
        self.assertEqual(page_replacement(refs, 3, "fifo")[0], 9)
        self.assertEqual(page_replacement(refs, 4, "fifo")[0], 10)

    def test_optimal_is_lower_bound(self):
        rng = random.Random(0)
        for _ in range(500):
            refs = [rng.randint(0, 7) for _ in range(rng.randint(5, 25))]
            f = rng.randint(1, 5)
            opt = page_replacement(refs, f, "optimal")[0]
            self.assertLessEqual(opt, page_replacement(refs, f, "fifo")[0])
            self.assertLessEqual(opt, page_replacement(refs, f, "lru")[0])
            self.assertGreaterEqual(opt, len(set(refs)) if f >= len(set(refs)) else f)


class TestBankers(unittest.TestCase):
    ALLOC = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
    MAX = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
    AVAIL = [3, 3, 2]

    def test_galvin_safe(self):
        safe, seq, _ = bankers_safety(self.ALLOC, self.MAX, self.AVAIL)
        self.assertTrue(safe)
        self.assertEqual(seq, [1, 3, 4, 0, 2])  # <P1, P3, P4, P0, P2>

    def test_verifier_accepts_other_valid_sequence(self):
        self.assertTrue(is_safe_sequence(self.ALLOC, self.MAX, self.AVAIL, [1, 3, 0, 2, 4]))
        self.assertFalse(is_safe_sequence(self.ALLOC, self.MAX, self.AVAIL, [0, 1, 2, 3, 4]))
        self.assertFalse(is_safe_sequence(self.ALLOC, self.MAX, self.AVAIL, [1, 3, 4]))

    def test_unsafe(self):
        safe, _, _ = bankers_safety(self.ALLOC, self.MAX, [0, 0, 0])
        self.assertFalse(safe)


class TestDisk(unittest.TestCase):
    Q, HEAD = [98, 183, 37, 122, 14, 124, 65, 67], 53

    def mv(self, algo, direction="up"):
        return head_movement(disk_schedule(self.HEAD, self.Q, algo, direction))

    def test_galvin(self):
        self.assertEqual(self.mv("fcfs"), 640)
        self.assertEqual(self.mv("sstf"), 236)
        self.assertEqual(self.mv("scan", "down"), 236)   # 53 -> 0 -> 183
        self.assertEqual(self.mv("scan", "up"), 331)     # 53 -> 199 -> 14
        self.assertEqual(self.mv("cscan", "up"), 382)    # 146 + 199 + 37
        self.assertEqual(self.mv("look", "up"), 299)     # 130 + 169
        self.assertEqual(self.mv("clook", "up"), 322)    # 130 + 169 + 23
        self.assertEqual(self.mv("cscan", "down"), 386)  # 53 + 199 + 134

    def test_every_request_served(self):
        for algo in os_gen.DISK_NAMES:
            for d in ("up", "down"):
                path = disk_schedule(self.HEAD, self.Q, algo, d)
                self.assertTrue(set(self.Q) <= set(path), (algo, d))


class TestGeneratorAndGrading(unittest.TestCase):
    def test_solution_passes_own_grader(self):
        """Every generated reference solution must be graded correct, so the
        'Final answer' line and the computed values always agree."""
        for split in ("train", "test"):
            for rec in os_gen.generate(3000, seed=7, split=split):
                self.assertTrue(grade(rec["final"], rec["messages"][-1]["content"]), rec["id"])

    def test_grader_rejects_wrong_answers(self):
        final = {"type": "numbers", "values": [10 / 3, 26 / 3]}
        self.assertTrue(grade(final, "... Final answer: Average WT = 3.33, Average TAT = 8.67"))
        self.assertTrue(grade(final, "Final answer: avg waiting time 10/3, avg turnaround 26/3"))
        self.assertFalse(grade(final, "Final answer: Average WT = 3.5, Average TAT = 8.67"))
        self.assertFalse(grade(final, "Final answer: Average TAT = 8.67, Average WT = 3.33"))

    def test_bankers_grading(self):
        final = {"type": "bankers", "safe": True, "alloc": TestBankers.ALLOC,
                 "max": TestBankers.MAX, "avail": TestBankers.AVAIL}
        self.assertTrue(grade(final, "Final answer: Safe; safe sequence = <P1, P3, P0, P2, P4>"))
        self.assertFalse(grade(final, "Final answer: Safe; <P0, P1, P2, P3, P4>"))
        self.assertFalse(grade(final, "Final answer: The system is unsafe"))

    def test_deterministic_and_unique(self):
        a = os_gen.generate(200, seed=1)
        b = os_gen.generate(200, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(len({r["messages"][1]["content"] for r in a}), 200)


if __name__ == "__main__":
    unittest.main()
