// Turn LaTeX math in model answers into plain readable text: "$$ z = \sum_i w_i x_i + b $$" -> "z = Σ wᵢxᵢ + b".
// Mirrors data/plain_math.py, which cleans the training data the same way.
(function (root) {
  const SYMBOLS = {
    cdot: "·", times: "×", div: "÷", pm: "±", mp: "∓", geq: "≥", ge: "≥", leq: "≤", le: "≤", neq: "≠", ne: "≠",
    approx: "≈", equiv: "≡", sim: "~", propto: "∝", infty: "∞", sum: "Σ", prod: "Π", int: "∫", partial: "∂", nabla: "∇",
    rightarrow: "→", to: "→", leftarrow: "←", Rightarrow: "⇒", implies: "⇒", Leftarrow: "⇐", leftrightarrow: "↔",
    Leftrightarrow: "⇔", iff: "⇔", mapsto: "↦", in: "∈", notin: "∉", subset: "⊂", subseteq: "⊆", supset: "⊃",
    supseteq: "⊇", cup: "∪", cap: "∩", emptyset: "∅", varnothing: "∅", forall: "∀", exists: "∃", neg: "¬", lnot: "¬",
    land: "∧", wedge: "∧", lor: "∨", vee: "∨", oplus: "⊕", otimes: "⊗", circ: "∘", ldots: "…", cdots: "⋯", dots: "…",
    lfloor: "⌊", rfloor: "⌋", lceil: "⌈", rceil: "⌉", langle: "⟨", rangle: "⟩", mid: "|", vert: "|", lvert: "|",
    rvert: "|", Vert: "‖", log: "log", ln: "ln", exp: "exp", max: "max", min: "min", sin: "sin", cos: "cos", tan: "tan",
    lim: "lim", arg: "arg", argmax: "argmax", argmin: "argmin", degree: "°", prime: "′", star: "*", ast: "*",
    alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", zeta: "ζ", eta: "η", theta: "θ",
    iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ", sigma: "σ", tau: "τ",
    phi: "φ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω", Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ",
    Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  };
  const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉", "+": "₊", "-": "₋",
    "=": "₌", "(": "₍", ")": "₎", a: "ₐ", e: "ₑ", h: "ₕ", i: "ᵢ", j: "ⱼ", k: "ₖ", l: "ₗ", m: "ₘ", n: "ₙ", o: "ₒ",
    p: "ₚ", r: "ᵣ", s: "ₛ", t: "ₜ", u: "ᵤ", v: "ᵥ", x: "ₓ" };
  const SUP = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹", "+": "⁺", "-": "⁻",
    "=": "⁼", "(": "⁽", ")": "⁾", n: "ⁿ", i: "ⁱ", T: "ᵀ", "*": "*", "′": "′" };

  // Read one argument after a command: {...} (balanced) or a single character.
  function arg(s, i) {
    while (s[i] === " ") i++;
    if (s[i] !== "{") return [s[i] ?? "", i + 1];
    let depth = 0;
    for (let j = i; j < s.length; j++) {
      if (s[j] === "{") depth++;
      else if (s[j] === "}" && --depth === 0) return [s.slice(i + 1, j), j + 1];
    }
    return [s.slice(i + 1), s.length];
  }
  const simple = (x) => !/[\s+\-=·×÷±\/]/.test(x);     // a single term needs no brackets
  const PREFIX = new Set(["∂", "∇", "√", "¬", "Σ", "Π"]); // operators written tight against what follows
  const wrap = (x) => (simple(x) ? x : `(${x})`);
  function script(x, table, mark) {
    if ([...x].every((c) => table[c])) return [...x].map((c) => table[c]).join("");
    return mark + wrap(x);
  }

  // Matrices become [a, b; c, d]; aligned/cases rows go on separate lines.
  const ENV = /\\begin\{(\w+\*?)\}(?:\{[^}]*\})?([\s\S]*?)\\end\{\1\}/;
  function environment(m, name, body) {
    const rows = body.split(/\\\\/).filter((r) => r.trim());
    if (name.endsWith("matrix") || name === "array")
      return "[" + rows.map((r) => r.split("&").map(convert).join(", ")).join("; ") + "]";
    return rows.map((r) => r.split("&").map(convert).join(" ")).join("\n");
  }

  function convert(s) {
    while (ENV.test(s)) s = s.replace(new RegExp(ENV.source, "g"), environment);  // repeat for nested ones
    let out = "";
    for (let i = 0; i < s.length; ) {
      const c = s[i];
      if (c === "\\") {
        const m = /^\\([a-zA-Z]+|.)/.exec(s.slice(i));
        const name = m[1]; i += m[0].length;
        if (name === "frac" || name === "dfrac" || name === "tfrac") {
          const [a, i1] = arg(s, i); const [b, i2] = arg(s, i1); i = i2;
          out += `${wrap(convert(a))}/${wrap(convert(b))}`;
        } else if (name === "sqrt") {
          const [a, i1] = arg(s, i); i = i1; out += `√${wrap(convert(a))}`;
        } else if (name === "boxed") {
          const [a, i1] = arg(s, i); i = i1; out += convert(a);
        } else if (/^(text|textbf|textit|mathrm|mathbf|mathit|mathcal|mathbb|operatorname|bm|boldsymbol|hat|bar|vec|tilde|overline|underline)$/.test(name)) {
          const [a, i1] = arg(s, i); i = i1;
          const t = name === "text" ? a : convert(a);
          out += name === "hat" ? `${t}̂` : name === "bar" || name === "overline" ? `${t}̄` : name === "vec" ? `${t}⃗` : t;
        } else if (name === "binom") {
          const [a, i1] = arg(s, i); const [b, i2] = arg(s, i1); i = i2; out += `C(${convert(a)}, ${convert(b)})`;
        } else if (name === "left" || name === "right" || name === "big" || name === "Big" || name === "bigg" || name === "Bigg") {
          // size commands: keep the delimiter that follows, drop "." placeholders
          if (s[i] === ".") i++;
        } else if (name === "," || name === ";" || name === ":" || name === " " || name === "quad" || name === "qquad") {
          out += " ";
        } else if (name === "!") {
          // negative space
        } else if (name === "\\") {
          out += "\n";
        } else if (name === "{" || name === "}" || name === "%" || name === "$" || name === "&" || name === "_" || name === "#") {
          out += name;
        } else if (SYMBOLS[name] != null) {
          const sym = SYMBOLS[name];
          out += /^[a-z]+$/.test(sym) && /[\w)]$/.test(out) ? " " + sym : sym;
          if (/^[a-z]+$/.test(sym) && /^[A-Za-z0-9(]/.test(s.slice(i).trimStart())) out += " ";
          if (PREFIX.has(sym)) while (s[i] === " ") i++;
        } else {
          out += name;  // unknown command: keep its name rather than lose content
        }
      } else if (c === "_" || c === "^") {
        const [a, i1] = arg(s, i + 1); i = i1;
        out += script(convert(a), c === "_" ? SUB : SUP, c);
      } else if (c === "{" || c === "}") {
        i++;
      } else if (c === "&") {
        i++; out += " ";
      } else if (c === "~") {
        i++; out += " ";
      } else {
        out += c; i++;
      }
    }
    return out.replace(/[ \t]{2,}/g, " ").trim();
  }

  // Display math, \[...\], \(...\) and inline $...$ (not prices like "$5 and $10").
  const MATH = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^\s$](?:[^$\n]*?[^\s$])?)\$/g;

  function plainMath(text, onMath) {
    return text.replace(MATH, (m, d1, d2, i1, i2) => {
      const display = d1 != null || d2 != null;
      const plain = convert(d1 ?? d2 ?? i1 ?? i2);
      return onMath ? onMath(plain, display) : display ? `\n${plain}\n` : plain;
    }).replace(/\n{3,}/g, "\n\n");
  }

  root.plainMath = plainMath;
  root.latexToText = convert;
  if (typeof module !== "undefined") module.exports = { plainMath, latexToText: convert };
})(typeof window !== "undefined" ? window : globalThis);
