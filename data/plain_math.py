"""Turn LaTeX math into plain readable text: "$$ z = \\sum_i w_i x_i + b $$" -> "z = Σᵢ wᵢxᵢ + b".

Python twin of demo/static/plainmath.js (the chat app uses that one). Used to clean training answers so
the model learns to write math as plain text, and to show the judge what a student sees in the app.
"""
from __future__ import annotations

import re

SYMBOLS = {
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓", "geq": "≥", "ge": "≥", "leq": "≤", "le": "≤",
    "neq": "≠", "ne": "≠", "approx": "≈", "equiv": "≡", "sim": "~", "propto": "∝", "infty": "∞", "sum": "Σ",
    "prod": "Π", "int": "∫", "partial": "∂", "nabla": "∇", "rightarrow": "→", "to": "→", "leftarrow": "←",
    "Rightarrow": "⇒", "implies": "⇒", "Leftarrow": "⇐", "leftrightarrow": "↔", "Leftrightarrow": "⇔", "iff": "⇔",
    "mapsto": "↦", "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅", "forall": "∀", "exists": "∃", "neg": "¬",
    "lnot": "¬", "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨", "oplus": "⊕", "otimes": "⊗", "circ": "∘",
    "ldots": "…", "cdots": "⋯", "dots": "…", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "langle": "⟨", "rangle": "⟩", "mid": "|", "vert": "|", "lvert": "|", "rvert": "|", "Vert": "‖", "log": "log",
    "ln": "ln", "exp": "exp", "max": "max", "min": "min", "sin": "sin", "cos": "cos", "tan": "tan", "lim": "lim",
    "arg": "arg", "argmax": "argmax", "argmin": "argmin", "degree": "°", "prime": "′", "star": "*", "ast": "*",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",
}
SUB = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
SUP = dict(zip("0123456789+-=()niT", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵀ"))
SUP.update({"*": "*", "′": "′"})
PREFIX = {"∂", "∇", "√", "¬", "Σ", "Π"}  # operators written tight against what follows
WRAPPERS = {"text", "textbf", "textit", "mathrm", "mathbf", "mathit", "mathcal", "mathbb", "operatorname", "bm",
            "boldsymbol", "hat", "bar", "vec", "tilde", "overline", "underline"}
MATH = re.compile(r"\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^\s$](?:[^$\n]*?[^\s$])?)\$")


def _arg(s: str, i: int) -> tuple[str, int]:
    """One argument after a command: {...} (balanced) or a single character."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] != "{":
        return s[i], i + 1
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def _simple(x: str) -> bool:
    return not re.search(r"[\s+\-=·×÷±/]", x)


def _wrap(x: str) -> str:
    return x if _simple(x) else f"({x})"


def _script(x: str, table: dict, mark: str) -> str:
    if x and all(c in table for c in x):
        return "".join(table[c] for c in x)
    return mark + _wrap(x)


def latex_to_text(s: str) -> str:
    out = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+|.)", s[i:], re.S)
            if not m:
                i += 1
                continue
            name = m.group(1)
            i += len(m.group(0))
            if name in ("frac", "dfrac", "tfrac"):
                a, i = _arg(s, i)
                b, i = _arg(s, i)
                out += f"{_wrap(latex_to_text(a))}/{_wrap(latex_to_text(b))}"
            elif name == "sqrt":
                a, i = _arg(s, i)
                out += f"√{_wrap(latex_to_text(a))}"
            elif name == "boxed":
                a, i = _arg(s, i)
                out += latex_to_text(a)
            elif name in WRAPPERS:
                a, i = _arg(s, i)
                t = a if name == "text" else latex_to_text(a)
                out += {"hat": t + "\u0302", "bar": t + "\u0304", "overline": t + "\u0304", "vec": t + "\u20d7"}.get(name, t)
            elif name == "binom":
                a, i = _arg(s, i)
                b, i = _arg(s, i)
                out += f"C({latex_to_text(a)}, {latex_to_text(b)})"
            elif name in ("left", "right", "big", "Big", "bigg", "Bigg"):
                if i < len(s) and s[i] == ".":
                    i += 1
            elif name in (",", ";", ":", " ", "quad", "qquad"):
                out += " "
            elif name == "!":
                pass
            elif name == "\\":
                out += "\n"
            elif name in "{}%$&_#":
                out += name
            elif name in SYMBOLS:
                sym = SYMBOLS[name]
                word = sym.isalpha() and sym.isascii()
                out += " " + sym if word and re.search(r"[\w)]$", out) else sym
                if word and re.match(r"[A-Za-z0-9(]", s[i:].lstrip()):
                    out += " "
                if sym in PREFIX:
                    while i < len(s) and s[i] == " ":
                        i += 1
            else:
                out += name  # unknown command: keep its name rather than lose content
        elif c in "_^":
            a, i = _arg(s, i + 1)
            out += _script(latex_to_text(a), SUB if c == "_" else SUP, c)
        elif c in "{}":
            i += 1
        elif c in "&~":
            out += " "
            i += 1
        else:
            out += c
            i += 1
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def plain_math(text: str) -> str:
    """Replace every math span in `text` with plain text; display math gets its own line."""
    def repl(m: re.Match) -> str:
        display = m.group(1) is not None or m.group(2) is not None
        plain = latex_to_text(next(g for g in m.groups() if g is not None))
        return f"\n{plain}\n" if display else plain
    return re.sub(r"\n{3,}", "\n\n", MATH.sub(repl, text))


def has_latex(text: str) -> bool:
    return bool(MATH.search(text)) or bool(re.search(r"\\(frac|sum|cdot|times|sqrt|text|mathbf|boxed)\b", text))
