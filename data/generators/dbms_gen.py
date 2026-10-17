"""Verified DBMS (functional dependency theory) problem generator.

Problem families (answers computed by algorithm):
  * Attribute closure X+
  * All candidate keys
  * Highest normal form (1NF / 2NF / 3NF / BCNF)
  * Lossless-join test for a binary decomposition

Relations are R(A, B, C, ...) and every relation is assumed to be in 1NF (atomic values).

Usage:
  python -m data.generators.dbms_gen --n 600 --seed 42 --out data/processed/gen_dbms_train.jsonl
"""
from __future__ import annotations

import random
from itertools import combinations

from .common import Problem, cli, run_generators

FD = tuple[frozenset, frozenset]

# =====================================================================
# Functional dependency algorithms
# =====================================================================


def closure(attrs: frozenset, fds: list[FD]) -> tuple[frozenset, list[str]]:
    """Attribute closure with the order in which FDs fired (for the explanation)."""
    result, trace = set(attrs), []
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if lhs <= result and not rhs <= result:
                added = rhs - result
                result |= rhs
                trace.append(f"{s(lhs)} → {s(rhs)} applies, so add {s(added)} → {{{', '.join(sorted(result))}}}")
                changed = True
    return frozenset(result), trace


def candidate_keys(attrs: frozenset, fds: list[FD]) -> list[frozenset]:
    keys: list[frozenset] = []
    for size in range(1, len(attrs) + 1):
        for combo in combinations(sorted(attrs), size):
            c = frozenset(combo)
            if any(k <= c for k in keys):
                continue
            if closure(c, fds)[0] == attrs:
                keys.append(c)
    return keys


def normal_form(attrs: frozenset, fds: list[FD]) -> tuple[str, str]:
    """Highest normal form of R(attrs) under fds, with the violating dependency explained."""
    keys = candidate_keys(attrs, fds)
    prime = frozenset().union(*keys)
    # 2NF: no non-prime attribute depends on a proper subset of a candidate key.
    for k in keys:
        for size in range(1, len(k)):
            for sub in combinations(sorted(k), size):
                sub = frozenset(sub)
                extra = (closure(sub, fds)[0] - sub) - prime
                if extra:
                    return "1NF", (f"{s(sub)} is a proper subset of candidate key {s(k)} and determines non-prime "
                                   f"attribute(s) {s(extra)}: a partial dependency, so R is not in 2NF.")
    # 3NF: for every X → A, X is a superkey or A is prime.
    for lhs, rhs in fds:
        for a in sorted(rhs - lhs):
            if closure(lhs, fds)[0] != attrs and a not in prime:
                return "2NF", (f"{s(lhs)} → {a}: {s(lhs)} is not a superkey and {a} is non-prime "
                               "(a transitive dependency), so R is not in 3NF.")
    # BCNF: for every non-trivial X → Y, X is a superkey.
    for lhs, rhs in fds:
        if rhs - lhs and closure(lhs, fds)[0] != attrs:
            return "3NF", (f"{s(lhs)} → {s(rhs - lhs)}: {s(lhs)} is not a superkey, so R is not in BCNF; but every "
                           "attribute it determines is prime, so 3NF holds.")
    return "BCNF", "The left side of every non-trivial FD is a superkey, so R is in BCNF."


def s(x) -> str:
    """Compact attribute-set name: frozenset({'B', 'A'}) -> 'AB'."""
    return "".join(sorted(x))


def fd_str(fds: list[FD]) -> str:
    return "{" + ", ".join(f"{s(l)} → {s(r)}" for l, r in fds) + "}"


def random_schema(rng: random.Random, split: str) -> tuple[frozenset, list[FD]]:
    n = rng.randint(4, 6) if split == "train" else rng.randint(5, 7)
    attrs = [chr(ord("A") + i) for i in range(n)]
    n_fd = rng.randint(2, 4) if split == "train" else rng.randint(3, 5)
    fds: list[FD] = []
    while len(fds) < n_fd:
        lhs = frozenset(rng.sample(attrs, rng.choice([1, 1, 2])))
        rest = [a for a in attrs if a not in lhs]
        rhs = frozenset(rng.sample(rest, rng.choice([1, 1, 2])))
        if (lhs, rhs) not in fds:
            fds.append((lhs, rhs))
    return frozenset(attrs), fds


def _relation(attrs) -> str:
    return f"R({', '.join(sorted(attrs))})"


def _keys_reasoning(attrs: frozenset, fds: list[FD], keys: list[frozenset]) -> str:
    rhs_all = frozenset().union(*(r for _, r in fds))
    must = attrs - rhs_all
    lines = [f"Attributes that never appear on a right-hand side must be in every key: {{{', '.join(sorted(must))}}}."
             if must else "Every attribute appears on some right-hand side, so no attribute is forced into every key."]
    if must:
        c, _ = closure(must, fds)
        lines.append(f"{s(must)}+ = {{{', '.join(sorted(c))}}}" + (" = R, so it is the only candidate key."
                                                                     if c == attrs else ", which is not all of R."))
    if not (must and closure(must, fds)[0] == attrs):  # otherwise the line above already settles it
        for k in keys:
            lines.append(f"{s(k)}+ = {{{', '.join(sorted(attrs))}}} = R, and no proper subset of {s(k)} is a superkey.")
    return "\n".join(f"- {line}" for line in lines)


# =====================================================================
# Problem families
# =====================================================================

def gen_closure(rng: random.Random, split: str = "train") -> Problem:
    while True:
        attrs, fds = random_schema(rng, split)
        x = frozenset(rng.sample(sorted(attrs), rng.choice([1, 1, 2])))
        c, trace = closure(x, fds)
        if len(c) > len(x):  # make sure at least one FD fires
            break
    q = rng.choice([
        f"Given relation {_relation(attrs)} with functional dependencies {fd_str(fds)}, compute the closure {s(x)}+.",
        f"For the functional dependencies {fd_str(fds)} on {_relation(attrs)}, find the attribute closure of {{{', '.join(sorted(x))}}}. "
        f"Is {s(x)} a superkey?",
    ])
    is_key = c == attrs
    sol = (f"Start with {s(x)}+ = {{{', '.join(sorted(x))}}} and repeatedly apply FDs whose left side is contained "
           "in the current set:\n\n" + "\n".join(f"- {t}" for t in trace)
           + f"\n\nNo more FDs add attributes, so {s(x)}+ = **{{{', '.join(sorted(c))}}}**. "
           + (f"It contains every attribute, so {s(x)} **is** a superkey." if is_key
              else f"It does not contain all of R, so {s(x)} is **not** a superkey.")
           + f"\n\n**Final answer:** {s(x)}+ = {{{', '.join(sorted(c))}}}")
    return Problem("DBMS", "fd/closure", q, sol,
                   {"type": "attrset", "values": sorted(c), "attrs": sorted(attrs)})


def gen_keys(rng: random.Random, split: str = "train") -> Problem:
    while True:
        attrs, fds = random_schema(rng, split)
        keys = candidate_keys(attrs, fds)
        if all(len(k) < len(attrs) for k in keys):
            break
    q = rng.choice([
        f"Find all candidate keys of {_relation(attrs)} with functional dependencies {fd_str(fds)}.",
        f"Relation {_relation(attrs)} satisfies the functional dependencies {fd_str(fds)}. List every candidate key and the prime attributes.",
    ])
    prime = frozenset().union(*keys)
    sol = (_keys_reasoning(attrs, fds, keys)
           + f"\n\nCandidate keys: **{', '.join(s(k) for k in keys)}**. Prime attributes: {{{', '.join(sorted(prime))}}}."
           + f"\n\n**Final answer:** Candidate keys = {', '.join('{' + ', '.join(sorted(k)) + '}' for k in keys)}")
    return Problem("DBMS", "fd/candidate_keys", q, sol,
                   {"type": "keysets", "values": [sorted(k) for k in keys], "attrs": sorted(attrs)})


NF_ORDER = ["1NF", "2NF", "3NF", "BCNF"]


def gen_normal_form(rng: random.Random, split: str = "train") -> Problem:
    target = rng.choice(NF_ORDER)  # balance the answer distribution
    for _ in range(500):
        attrs, fds = random_schema(rng, split)
        nf, why = normal_form(attrs, fds)
        if nf == target:
            break
    keys = candidate_keys(attrs, fds)
    prime = frozenset().union(*keys)
    q = rng.choice([
        f"Relation {_relation(attrs)} has functional dependencies {fd_str(fds)}. Assuming all attributes are "
        "atomic, what is the highest normal form R satisfies (1NF, 2NF, 3NF or BCNF)? Justify.",
        f"Determine the highest normal form of {_relation(attrs)} under the functional dependencies {fd_str(fds)} (all values atomic).",
    ])
    sol = ("**Step 1: candidate keys.**\n\n" + _keys_reasoning(attrs, fds, keys)
           + f"\n\nCandidate keys: {', '.join(s(k) for k in keys)}; prime attributes: {{{', '.join(sorted(prime))}}}; "
           f"non-prime: {('{' + ', '.join(sorted(attrs - prime)) + '}') if attrs - prime else '∅'}.\n\n"
           "**Step 2: test 2NF → 3NF → BCNF.**\n\n" + why
           + f"\n\n**Final answer:** {nf}")
    return Problem("DBMS", "normalization/highest_nf", q, sol,
                   {"type": "choice", "value": nf, "options": NF_ORDER})


def gen_lossless(rng: random.Random, split: str = "train") -> Problem:
    want_lossless = rng.random() < 0.5  # balance the answer distribution
    for _ in range(500):
        attrs, fds = random_schema(rng, split)
        order = sorted(attrs)
        common = frozenset(rng.sample(order, rng.choice([1, 1, 2])))
        rest = [a for a in order if a not in common]
        rng.shuffle(rest)
        cut = rng.randint(1, len(rest) - 1)
        r1, r2 = common | frozenset(rest[:cut]), common | frozenset(rest[cut:])
        c, _ = closure(common, fds)
        lossless = r1 <= c or r2 <= c
        if lossless == want_lossless:
            break
    q = (f"{_relation(attrs)} with functional dependencies {fd_str(fds)} is decomposed into R1({', '.join(sorted(r1))}) and "
         f"R2({', '.join(sorted(r2))}). Is the decomposition lossless or lossy?")
    sol = (f"A binary decomposition is lossless iff R1 ∩ R2 → R1 or R1 ∩ R2 → R2.\n\n"
           f"R1 ∩ R2 = {{{', '.join(sorted(common))}}}; ({s(common)})+ = {{{', '.join(sorted(c))}}}.\n\n"
           + (f"({s(common)})+ contains all of {'R1' if r1 <= c else 'R2'}, so the common attributes form a key of "
              f"{'R1' if r1 <= c else 'R2'} and the join is **lossless**." if lossless else
              f"({s(common)})+ contains neither R1 nor R2, so joining R1 and R2 can create spurious tuples: the "
              "decomposition is **lossy**.")
           + f"\n\n**Final answer:** {'Lossless' if lossless else 'Lossy'}")
    return Problem("DBMS", "decomposition/lossless", q, sol,
                   {"type": "choice", "value": "lossless" if lossless else "lossy", "options": ["lossless", "lossy"]})


GENERATORS = [(gen_closure, 0.25), (gen_keys, 0.30), (gen_normal_form, 0.30), (gen_lossless, 0.15)]


def generate(n: int, seed: int, split: str = "train") -> list[dict]:
    return run_generators(GENERATORS, "dbms", n, seed, split)


if __name__ == "__main__":
    cli(GENERATORS, "dbms", "DBMS", default_n=600)
