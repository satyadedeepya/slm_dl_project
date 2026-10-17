"""Grade a model's free-text answer against a generator problem's `final` spec.

Only the text after the last "Final answer" marker is graded.

Spec types:
  numbers  {"values": [...], "rel_tol": 0.01?}
           every expected value must appear, in order; absolute tolerance 0.011,
           or a relative tolerance when "rel_tol" is given.
  tokens   {"values": ["192.168.1.128", "0110", ...]}
           every token must appear as a whole token (not inside a longer
           number/bit string); case-insensitive, "0x" prefixes ignored.
  bankers  {"safe": bool, "alloc", "max", "avail"}
           verdict must match; a stated safe sequence is checked by the safety
           verifier, so any valid safe sequence is accepted.
  attrset  {"values": ["A", "B", ...], "attrs": [...]}
           the last {...} set (or the letters after the last '=') must equal the set.
  keysets  {"values": [["A","B"], ["C"]], "attrs": [...]}
           the set of candidate keys stated must equal the expected set of keys.
  choice   {"value": "3NF", "options": ["1NF", "2NF", "3NF", "BCNF"]}
           the option mentioned last must be the expected one.
"""
from __future__ import annotations

import re

from .common import final_segment, parse_numbers
from .os_gen import is_safe_sequence

TOL = 0.011  # accepts 8.67 / 8.666 for 26/3, rejects 8.6


def _numbers_in_order(expected: list[float], found: list[float], rel_tol: float | None) -> bool:
    i = 0
    for v in found:
        if i < len(expected):
            tol = rel_tol * abs(expected[i]) if rel_tol else TOL
            if abs(v - expected[i]) <= max(tol, 1e-9):
                i += 1
    return i == len(expected)


def _has_token(seg: str, token: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(token.lower())}(?![\w]|\.\d)", seg) is not None


def _letters(text: str, attrs: set[str]) -> set[str]:
    return {c for c in re.findall(r"[A-Z]", text) if c in attrs}


def _strip_marker(seg: str) -> str:
    return re.sub(r"(?i)^\**\s*final answer\s*\**\s*:?\s*\**", "", seg)


def _attrset(seg: str, attrs: set[str]) -> set[str]:
    seg = _strip_marker(seg)
    groups = re.findall(r"\{([^}]*)\}", seg)
    if groups:
        return _letters(groups[-1], attrs)
    tail = re.split(r"[=:]", seg)[-1]
    return _letters(tail, attrs)


def _keysets(seg: str, attrs: set[str]) -> set[frozenset]:
    seg = _strip_marker(seg)
    groups = re.findall(r"[{(]([^})]*)[})]", seg)
    if not groups:  # e.g. "Candidate keys: AB, BC"
        tail = re.split(r":", seg, maxsplit=1)[-1]
        groups = re.split(r",|\band\b|;", tail)
    keys = {frozenset(_letters(g, attrs)) for g in groups}
    return {k for k in keys if k}


def _last_option(seg: str, options: list[str]) -> str | None:
    best, pos = None, -1
    low = seg.lower()
    for opt in options:
        for m in re.finditer(rf"(?<![\w]){re.escape(opt.lower())}(?![\w])", low):
            if m.start() > pos:
                best, pos = opt, m.start()
    return best


def grade(final: dict, output: str) -> bool:
    seg = final_segment(output)
    kind = final["type"]
    if kind == "numbers":
        return _numbers_in_order([float(v) for v in final["values"]], parse_numbers(seg), final.get("rel_tol"))
    if kind == "tokens":
        low = seg.lower().replace("0x", "")
        return all(_has_token(low, str(t).lower().replace("0x", "")) for t in final["values"])
    if kind == "bankers":
        low = seg.lower()
        says_unsafe = "unsafe" in low or "not safe" in low or "not in a safe" in low
        if not final["safe"]:
            return says_unsafe
        if says_unsafe:
            return False
        seq = [int(x) for x in re.findall(r"P\s*(\d+)", seg)]
        return is_safe_sequence(final["alloc"], final["max"], final["avail"], seq)
    if kind == "attrset":
        return _attrset(seg, set(final["attrs"])) == set(final["values"])
    if kind == "keysets":
        expected = {frozenset(k) for k in final["values"]}
        return _keysets(seg, set(final["attrs"])) == expected
    if kind == "choice":
        return _last_option(seg, final["options"]) == final["value"]
    raise ValueError(f"unknown final type {kind}")
