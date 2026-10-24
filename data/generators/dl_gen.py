"""Verified Deep Learning problem generator.

Problem families (answers computed, never written by an LLM):
  * Convolution: output shape and parameter count of a conv layer
  * Pooling: output shape of a pooling layer
  * MLP: number of trainable parameters of a fully connected network
  * Forward pass: a small 2-layer network with ReLU / sigmoid
  * Softmax + cross-entropy for one example
  * Gradient descent: one step for a linear neuron with squared-error loss
  * Backpropagation: gradient of a sigmoid neuron with squared-error loss
  * Momentum SGD: two update steps
  * Scaled dot-product attention weights for one query
  * Classification metrics from a confusion matrix

Usage:
  python -m data.generators.dl_gen --n 600 --seed 42 --out data/processed/gen_dl_train.jsonl
"""
from __future__ import annotations

import math
import random

from .common import Problem, cli, md_table, run_generators


def num(x: float, d: int = 4) -> str:
    """Round to d decimals and drop trailing zeros (no thousands separators, so the grader can parse it)."""
    s = f"{x:.{d}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


# =====================================================================
# Shapes and parameter counts
# =====================================================================

def gen_conv(rng: random.Random, split: str = "train") -> Problem:
    if split == "train":
        h = rng.choice([28, 32, 64, 96, 128, 224])
        c = rng.choice([1, 3, 16, 32, 64])
        f = rng.choice([8, 16, 32, 64, 128])
        k = rng.choice([3, 5, 7])
    else:
        h = rng.choice([30, 48, 100, 150, 227])
        c = rng.choice([6, 12, 24, 48])
        f = rng.choice([10, 20, 40, 96])
        k = rng.choice([3, 5, 11])
    s = rng.choice([1, 2])
    p = rng.choice([0, (k - 1) // 2])
    out = (h + 2 * p - k) // s + 1
    params = (k * k * c + 1) * f
    q = (f"A convolutional layer receives an input of size {h} × {h} × {c} (height × width × channels). "
         f"It has {f} filters of size {k} × {k}, stride {s} and padding {p}. Find the output size and the number "
         f"of trainable parameters (including biases).")
    sol = (f"Output size per spatial dimension: ⌊(N + 2P − K) / S⌋ + 1 = ⌊({h} + 2·{p} − {k}) / {s}⌋ + 1 = "
           f"⌊{h + 2 * p - k} / {s}⌋ + 1 = {out}.\n\n"
           f"The depth of the output equals the number of filters: {f}.\n\n"
           f"Each filter has K × K × C weights plus one bias: {k}·{k}·{c} + 1 = {k * k * c + 1}.\n"
           f"Parameters = ({k * k * c + 1}) × {f} = {params}.\n\n"
           f"**Final answer:** Output = {out} × {out} × {f}, parameters = {params}")
    return Problem("DL", "cnn/conv", q, sol, {"type": "numbers", "values": [out, out, f, params]})


def gen_pool(rng: random.Random, split: str = "train") -> Problem:
    h = rng.choice([16, 24, 32, 56, 112] if split == "train" else [20, 40, 60, 75])
    c = rng.choice([16, 32, 64] if split == "train" else [24, 48, 96])
    k, s = rng.choice([(2, 2), (3, 2), (3, 1)] if split == "train" else [(2, 2), (3, 3), (4, 2)])
    kind = rng.choice(["max", "average"])
    out = (h - k) // s + 1
    q = (f"A {kind}-pooling layer with window {k} × {k} and stride {s} (no padding) is applied to a feature map of "
         f"size {h} × {h} × {c}. What is the output size, and how many trainable parameters does the layer have?")
    sol = (f"Output size: ⌊(N − K) / S⌋ + 1 = ⌊({h} − {k}) / {s}⌋ + 1 = {out}. Pooling works on each channel "
           f"separately, so the depth stays {c}.\n\nPooling has no weights, so it has 0 trainable parameters.\n\n"
           f"**Final answer:** Output = {out} × {out} × {c}, parameters = 0")
    return Problem("DL", "cnn/pool", q, sol, {"type": "numbers", "values": [out, out, c, 0]})


def gen_mlp_params(rng: random.Random, split: str = "train") -> Problem:
    n_layers = rng.randint(2, 4)
    pool_in = [784, 100, 64, 20, 10] if split == "train" else [512, 300, 50, 12]
    pool_h = [512, 256, 128, 64, 32] if split == "train" else [400, 200, 100, 48, 24]
    sizes = [rng.choice(pool_in)] + [rng.choice(pool_h) for _ in range(n_layers - 1)] + [rng.choice([2, 3, 5, 10])]
    rows, total = [], 0
    for i in range(len(sizes) - 1):
        w, b = sizes[i] * sizes[i + 1], sizes[i + 1]
        total += w + b
        rows.append([f"{sizes[i]} → {sizes[i + 1]}", f"{sizes[i]}·{sizes[i + 1]} = {w}", b, w + b])
    arch = " → ".join(str(x) for x in sizes)
    q = (f"A fully connected network has layer sizes {arch} (input → hidden layers → output), with a bias on every "
         f"neuron. How many trainable parameters does it have?")
    sol = ("Each dense layer has (inputs × outputs) weights plus one bias per output.\n\n"
           + md_table(["Layer", "Weights", "Biases", "Total"], rows)
           + f"\n\nTotal = {' + '.join(str(r[3]) for r in rows)} = {total}.\n\n**Final answer:** {total} parameters")
    return Problem("DL", "mlp/params", q, sol, {"type": "numbers", "values": [total]})


# =====================================================================
# Forward pass, softmax, losses
# =====================================================================

def gen_forward(rng: random.Random, split: str = "train") -> Problem:
    vals = [-2, -1, 0, 1, 2, 3] if split == "train" else [-1.5, -0.5, 0.5, 1.5, 2.5]
    x = [rng.choice([1, 2, 3, -1]) if split == "train" else rng.choice([0.5, 1.5, -2, 4]) for _ in range(2)]
    W1 = [[rng.choice(vals) for _ in range(2)] for _ in range(2)]
    b1 = [rng.choice([-1, 0, 1]) for _ in range(2)]
    w2 = [rng.choice(vals) for _ in range(2)]
    b2 = rng.choice([-1, 0, 1])
    z1 = [W1[j][0] * x[0] + W1[j][1] * x[1] + b1[j] for j in range(2)]
    h = [max(0.0, z) for z in z1]
    z2 = w2[0] * h[0] + w2[1] * h[1] + b2
    y = sigmoid(z2)
    q = (f"A network has input x = ({num(x[0])}, {num(x[1])}), a hidden layer of two ReLU neurons with weights "
         f"W1 = [[{num(W1[0][0])}, {num(W1[0][1])}], [{num(W1[1][0])}, {num(W1[1][1])}]] (one row per neuron) and "
         f"biases b1 = ({b1[0]}, {b1[1]}), and one sigmoid output neuron with weights w2 = ({num(w2[0])}, {num(w2[1])}) "
         f"and bias b2 = {b2}. Compute the hidden activations, the output pre-activation z2 and the output ŷ.")
    sol = (f"Hidden pre-activations:\n- z1₁ = {num(W1[0][0])}·{num(x[0])} + {num(W1[0][1])}·{num(x[1])} + {b1[0]} = {num(z1[0])}\n"
           f"- z1₂ = {num(W1[1][0])}·{num(x[0])} + {num(W1[1][1])}·{num(x[1])} + {b1[1]} = {num(z1[1])}\n\n"
           f"ReLU: h = (max(0, {num(z1[0])}), max(0, {num(z1[1])})) = ({num(h[0])}, {num(h[1])})\n\n"
           f"Output: z2 = {num(w2[0])}·{num(h[0])} + {num(w2[1])}·{num(h[1])} + {b2} = {num(z2)}; "
           f"ŷ = σ({num(z2)}) = 1 / (1 + e^(−{num(z2)})) = {num(y)}\n\n"
           f"**Final answer:** h = ({num(h[0])}, {num(h[1])}), z2 = {num(z2)}, ŷ = {num(y)}")
    return Problem("DL", "forward/mlp", q, sol,
                   {"type": "numbers", "values": [round(h[0], 4), round(h[1], 4), round(z2, 4), round(y, 4)],
                    "rel_tol": 0.01})


def gen_softmax_ce(rng: random.Random, split: str = "train") -> Problem:
    pool = [-1, 0, 1, 2, 3] if split == "train" else [-0.5, 0.5, 1.5, 2.5]
    n = 3
    z = [rng.choice(pool) for _ in range(n)]
    t = rng.randrange(n)
    m = max(z)
    e = [math.exp(v - m) for v in z]
    p = [v / sum(e) for v in e]
    loss = -math.log(p[t])
    q = (f"A classifier outputs logits z = ({', '.join(num(v) for v in z)}) for classes 1–{n}. The true class is "
         f"class {t + 1}. Compute the softmax probabilities and the cross-entropy loss (natural log).")
    sol = (f"Softmax: pᵢ = e^(zᵢ) / Σⱼ e^(zⱼ). Subtracting the maximum logit ({num(m)}) keeps it numerically stable:\n"
           + "\n".join(f"- e^({num(v)} − {num(m)}) = {num(ev)}" for v, ev in zip(z, e))
           + f"\n\nSum = {num(sum(e))}, so p = ({', '.join(num(v) for v in p)}).\n\n"
           f"Cross-entropy = −ln(p_true) = −ln({num(p[t])}) = {num(loss)}.\n\n"
           f"**Final answer:** p = ({', '.join(num(v) for v in p)}), loss = {num(loss)}")
    return Problem("DL", "loss/softmax_ce", q, sol,
                   {"type": "numbers", "values": [round(v, 4) for v in p] + [round(loss, 4)], "rel_tol": 0.01})


# =====================================================================
# Gradients and optimisers
# =====================================================================

def gen_gd_linear(rng: random.Random, split: str = "train") -> Problem:
    w = rng.choice([0.5, 1, 1.5, 2, -1] if split == "train" else [0.2, 0.8, -0.4, 1.2])
    b = rng.choice([0, 0.5, -0.5] if split == "train" else [0.1, -0.3])
    x = rng.choice([1, 2, 3, -2] if split == "train" else [1.5, -1, 2.5])
    y = rng.choice([2, 4, 5, -1] if split == "train" else [3, 1.5, -2])
    lr = rng.choice([0.1, 0.05, 0.01] if split == "train" else [0.2, 0.02])
    yh = w * x + b
    err = yh - y
    gw, gb = 2 * err * x, 2 * err
    w2, b2 = w - lr * gw, b - lr * gb
    q = (f"A linear neuron computes ŷ = w·x + b with w = {num(w)} and b = {num(b)}. For one training example "
         f"x = {num(x)}, y = {num(y)} and loss L = (ŷ − y)², compute the gradients ∂L/∂w and ∂L/∂b and the "
         f"parameters after one gradient-descent step with learning rate {lr}.")
    sol = (f"Forward: ŷ = {num(w)}·{num(x)} + {num(b)} = {num(yh)}; error ŷ − y = {num(err)}.\n\n"
           f"∂L/∂w = 2(ŷ − y)·x = 2·{num(err)}·{num(x)} = {num(gw)}\n"
           f"∂L/∂b = 2(ŷ − y) = {num(gb)}\n\n"
           f"Update: w ← {num(w)} − {lr}·{num(gw)} = {num(w2)}; b ← {num(b)} − {lr}·{num(gb)} = {num(b2)}\n\n"
           f"**Final answer:** ∂L/∂w = {num(gw)}, ∂L/∂b = {num(gb)}, new w = {num(w2)}, new b = {num(b2)}")
    return Problem("DL", "optim/gd_linear", q, sol,
                   {"type": "numbers", "values": [round(gw, 4), round(gb, 4), round(w2, 4), round(b2, 4)],
                    "rel_tol": 0.01})


def gen_backprop_sigmoid(rng: random.Random, split: str = "train") -> Problem:
    pool = [-1, -0.5, 0.5, 1, 2] if split == "train" else [-1.5, 0.25, 0.75, 1.5]
    w1, w2 = rng.choice(pool), rng.choice(pool)
    b = rng.choice([0, 0.5, -0.5] if split == "train" else [0.2, -0.2])
    if split == "train":
        x1, x2 = rng.choice([1, 2, -1]), rng.choice([1, 0, 3])
    else:
        x1, x2 = rng.choice([0.5, -2]), rng.choice([1.5, 2])
    t = rng.choice([0, 1])
    z = w1 * x1 + w2 * x2 + b
    a = sigmoid(z)
    dz = (a - t) * a * (1 - a)
    g1, g2 = dz * x1, dz * x2
    q = (f"A sigmoid neuron computes a = σ(z) with z = w₁x₁ + w₂x₂ + b, where w₁ = {num(w1)}, w₂ = {num(w2)}, "
         f"b = {num(b)}. For input (x₁, x₂) = ({num(x1)}, {num(x2)}), target t = {t} and loss L = ½(a − t)², "
         f"use backpropagation to find ∂L/∂w₁ and ∂L/∂w₂.")
    sol = (f"Forward: z = {num(w1)}·{num(x1)} + {num(w2)}·{num(x2)} + {num(b)} = {num(z)}; a = σ({num(z)}) = {num(a)}.\n\n"
           f"Chain rule: ∂L/∂z = (a − t)·σ'(z) = (a − t)·a(1 − a) = ({num(a)} − {t})·{num(a)}·{num(1 - a)} = {num(dz, 6)}.\n\n"
           f"∂L/∂w₁ = ∂L/∂z · x₁ = {num(dz, 6)}·{num(x1)} = {num(g1, 6)}\n"
           f"∂L/∂w₂ = ∂L/∂z · x₂ = {num(dz, 6)}·{num(x2)} = {num(g2, 6)}\n\n"
           f"**Final answer:** ∂L/∂w₁ = {num(g1, 6)}, ∂L/∂w₂ = {num(g2, 6)}")
    return Problem("DL", "backprop/sigmoid", q, sol,
                   {"type": "numbers", "values": [round(g1, 6), round(g2, 6)], "rel_tol": 0.02})


def gen_momentum(rng: random.Random, split: str = "train") -> Problem:
    w = rng.choice([1, 2, -1, 0.5] if split == "train" else [1.5, -0.5, 3])
    lr = rng.choice([0.1, 0.05] if split == "train" else [0.2, 0.02])
    beta = rng.choice([0.9, 0.5] if split == "train" else [0.8, 0.6])
    g = [rng.choice([1, 2, -1, 0.5, 4]) if split == "train" else rng.choice([1.5, -2, 3, 0.25]) for _ in range(2)]
    w0, v, ws, lines = w, 0.0, [], []
    for i, gi in enumerate(g, 1):
        v_new = beta * v + gi
        w_new = w - lr * v_new
        lines.append(f"Step {i}: v = {beta}·{num(v)} + {num(gi)} = {num(v_new)}; "
                     f"w = {num(w)} − {lr}·{num(v_new)} = {num(w_new)}")
        v, w = v_new, w_new
        ws.append(w)
    q = (f"SGD with momentum uses v ← β·v + g and w ← w − η·v, starting from v = 0. With initial w = {num(w0)}, "
         f"η = {lr}, β = {beta} and gradients g₁ = {num(g[0])}, g₂ = {num(g[1])}, find w after each of the two steps.")
    sol = ("\n".join(lines)
           + f"\n\n**Final answer:** w after step 1 = {num(ws[0])}, w after step 2 = {num(ws[1])}")
    return Problem("DL", "optim/momentum", q, sol,
                   {"type": "numbers", "values": [round(x, 4) for x in ws], "rel_tol": 0.01})


# =====================================================================
# Attention and metrics
# =====================================================================

def gen_attention(rng: random.Random, split: str = "train") -> Problem:
    pool = [-1, 0, 1, 2] if split == "train" else [-2, 0.5, 1.5, 3]
    d = 2
    qv = [rng.choice(pool) for _ in range(d)]
    keys = [[rng.choice(pool) for _ in range(d)] for _ in range(3)]
    scores = [(qv[0] * k[0] + qv[1] * k[1]) / math.sqrt(d) for k in keys]
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    wts = [v / sum(e) for v in e]
    q = (f"In scaled dot-product attention with d_k = {d}, a query q = ({num(qv[0])}, {num(qv[1])}) attends to keys "
         + ", ".join(f"k{i + 1} = ({num(k[0])}, {num(k[1])})" for i, k in enumerate(keys))
         + ". Compute the attention weights softmax(q·kᵢ / √d_k).")
    sol = ("Scores q·kᵢ / √2:\n"
           + "\n".join(f"- s{i + 1} = ({num(qv[0])}·{num(k[0])} + {num(qv[1])}·{num(k[1])}) / 1.4142 = {num(s)}"
                       for i, (k, s) in enumerate(zip(keys, scores)))
           + f"\n\nSoftmax (subtracting the maximum score {num(m)}): exponentials "
           + ", ".join(num(v) for v in e) + f", sum {num(sum(e))}.\n\n"
           f"**Final answer:** weights = ({', '.join(num(v) for v in wts)})")
    return Problem("DL", "attention/weights", q, sol,
                   {"type": "numbers", "values": [round(v, 4) for v in wts], "rel_tol": 0.01})


def gen_metrics(rng: random.Random, split: str = "train") -> Problem:
    r = range(10, 90) if split == "train" else range(91, 160)
    tp, fp, fn, tn = (rng.choice(r) for _ in range(4))
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    acc = (tp + tn) / (tp + fp + fn + tn)
    q = (f"A binary classifier produced TP = {tp}, FP = {fp}, FN = {fn} and TN = {tn} on a test set. Compute "
         f"accuracy, precision, recall and F1 score.")
    sol = (f"Accuracy = (TP + TN) / total = {tp + tn} / {tp + fp + fn + tn} = {num(acc)}\n"
           f"Precision = TP / (TP + FP) = {tp} / {tp + fp} = {num(prec)}\n"
           f"Recall = TP / (TP + FN) = {tp} / {tp + fn} = {num(rec)}\n"
           f"F1 = 2PR / (P + R) = 2·{num(prec)}·{num(rec)} / ({num(prec)} + {num(rec)}) = {num(f1)}\n\n"
           f"**Final answer:** Accuracy = {num(acc)}, precision = {num(prec)}, recall = {num(rec)}, F1 = {num(f1)}")
    return Problem("DL", "metrics/prf", q, sol,
                   {"type": "numbers", "values": [round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)],
                    "rel_tol": 0.01})


# =====================================================================
# Driver
# =====================================================================

GENERATORS = [(gen_conv, 0.14), (gen_pool, 0.06), (gen_mlp_params, 0.10), (gen_forward, 0.11),
              (gen_softmax_ce, 0.11), (gen_gd_linear, 0.11), (gen_backprop_sigmoid, 0.11), (gen_momentum, 0.08),
              (gen_attention, 0.09), (gen_metrics, 0.09)]


def generate(n: int, seed: int, split: str = "train") -> list[dict]:
    return run_generators(GENERATORS, "dl", n, seed, split)


if __name__ == "__main__":
    cli(GENERATORS, "dl", "DL", default_n=600)
