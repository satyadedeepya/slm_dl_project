"""Grade a model's free-text answer against a generator problem's `final` spec.

Spec types:
  {"type": "numbers", "values": [...]}  - every expected value must appear, in order,
                                          in the text after the last "Final answer".
  {"type": "bankers", "safe": bool, ...} - verdict must match; if safe, the stated
                                          sequence is checked by the safety verifier,
                                          so any valid safe sequence is accepted.
"""
from __future__ import annotations

import re

from .common import final_segment, parse_numbers
from .os_gen import is_safe_sequence

TOL = 0.011  # accepts 8.67 / 8.666 for 26/3, rejects 8.6


def _numbers_in_order(expected: list[float], found: list[float]) -> bool:
    i = 0
    for v in found:
        if i < len(expected) and abs(v - expected[i]) <= TOL:
            i += 1
    return i == len(expected)


def grade(final: dict, output: str) -> bool:
    seg = final_segment(output)
    kind = final["type"]
    if kind == "numbers":
        return _numbers_in_order([float(v) for v in final["values"]], parse_numbers(seg))
    if kind == "bankers":
        low = seg.lower()
        says_unsafe = "unsafe" in low or "not safe" in low or "not in a safe" in low
        if not final["safe"]:
            return says_unsafe
        if says_unsafe:
            return False
        seq = [int(x) for x in re.findall(r"P\s*(\d+)", seg)]
        return is_safe_sequence(final["alloc"], final["max"], final["avail"], seq)
    raise ValueError(f"unknown final type {kind}")
