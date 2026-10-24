# Deep Learning Stage Report: FT-C

**Goal:** add Deep Learning as a sixth subject in about 5–6 hours in total, without retraining from scratch and without losing the earlier skills.

> Status: complete. FT-C trained in 70 min (66 steps, final validation loss 0.460) and was evaluated on the five most informative tasks.

## 1. Approach

Retraining the full mix with DL added would take about 5.5 h of training plus about 3 h of evaluation. Instead, **FT-C continues training FT-B's LoRA adapter** on a small DL stage that also contains replay data from the first-stage mix:

| Part | Examples | Source and quality gate |
|---|---|---|
| Verified DL problems | 600 | `data/generators/dl_gen.py`; answers computed by the generator and cross-checked with PyTorch in tests |
| DL concept Q&A | 307 | Gemini teacher over 36 DL syllabus topics (`data/concepts/topics.json`); questions and answers generated separately; MCQ/TF kept only when the answer agrees with the key |
| Replay | 1,200 | random sample of the first-stage training mix (all sources), against forgetting |
| **Total** | **2,107** | 0.61M tokens, 66 optimiser steps |

- The concept job finished all 36 topics (395 items) during training. The run used the 307 items (28 topics) that were ready when training started.
- **Training:** the same LoRA settings as FT-B (rank 16, α 32, effective batch 32, one epoch) with a lower learning rate (1e-4), since the run continues an adapter rather than starting a new one.
- **Validation sample:** 50 fresh DL problems plus 50 first-stage examples.

## 2. Verified DL generators

Ten problem types, each with computed answers and a typed grading spec:

| Type | What is computed |
|---|---|
| `cnn/conv` | output size ⌊(N + 2P − K)/S⌋ + 1 and parameters (K·K·C + 1)·F |
| `cnn/pool` | output size, 0 parameters |
| `mlp/params` | weights and biases per dense layer |
| `forward/mlp` | 2-layer ReLU network with sigmoid output |
| `loss/softmax_ce` | stable softmax and cross-entropy |
| `optim/gd_linear` | gradients and one gradient-descent step for a linear neuron |
| `backprop/sigmoid` | ∂L/∂w through a sigmoid neuron |
| `optim/momentum` | two momentum-SGD steps |
| `attention/weights` | scaled dot-product attention weights |
| `metrics/prf` | accuracy, precision, recall, F1 |

**Tests** (`tests/test_dl_gen.py`, 8 tests):
- Every reference solution passes the grader (600 problems).
- The train and test questions are disjoint.
- The AlexNet conv1 shape is checked by hand.
- Conv, MLP, softmax/cross-entropy, gradient descent, backprop and attention answers are **recomputed independently with PyTorch** (`nn.Conv2d`, `nn.Linear`, `cross_entropy`, autograd, `softmax`).

## 3. Results

FT-C was evaluated on five tasks:
- **DL numericals:** the new skill.
- **DBMS, CN and Spider:** where the first stage helped, to check those gains are kept.
- **MBPP:** to check coding ability.

CS-Bench, OS numericals (already at the floor) and GATE (n = 56) were skipped to save about 1.5 h.

| Task | n | B0 | FT-B | **FT-C** | p (FT-C vs FT-B) | p (FT-C vs B0) |
|---|---|---|---|---|---|---|
| DL numericals, exact match | 100 | 2.0 | 3.0 | **5.0** | 0.73 | 0.45 |
| DL numericals, value-level | 100 | 12.6 | 15.8 | **26.0** [19.8–32.2] | **0.013** | **0.001** |
| DBMS FD problems | 100 | 15.0 | 30.0 | **31.0** | 1.00 | **0.003** |
| CN numericals | 100 | 1.0 | 8.0 | 7.0 | 1.00 | 0.070 |
| Spider (execution) | 200 | 50.5 | 60.0 | 59.5 | 1.00 | **0.015** |
| MBPP pass@1 | 200 | 41.5 | 42.5 | 40.0 | 0.38 | 0.79 |

Notes on the table:
- Exact-match p-values are exact McNemar tests.
- The value-level row is the share of expected numbers that appear in the final answer, using the grader's tolerance. Its p-values are paired sign-flip permutation tests (`eval/partial_credit.py`).

**DL value-level accuracy by problem type (FT-B → FT-C, %):**

| Problem type | FT-B | FT-C |
|---|---|---|
| conv shapes | 22 | **47** |
| softmax-CE | 0 | **56** |
| metrics | 22 | **44** |
| MLP params | 0 | 18 |
| forward pass | 20 | 27 |
| pooling (n = 3) | 33 | 17 |
| gradient descent | 36 | **8** |
| momentum | 31 | 19 |
| attention | 0 | 0 |
| backprop | 0 | 0 |

![Accuracy per task](figures/accuracy.png)

## 4. Analysis

1. **The DL stage adds measurable DL skill.**
   - Value-level accuracy on held-out DL numericals rises from 15.8% to 26.0% (p = 0.013).
   - The model learned the formulas: conv output size and parameters, softmax-CE, precision/recall/F1.
2. **Exact match stays low (5%).**
   - Every number in a multi-part answer must be right, and FT-C slips on multi-digit arithmetic. For example, it writes 11·11·48 + 1 = 5675 where the answer is 5809.
   - This is the same procedural weakness found in the Week 4 error analysis.
3. **Some types got worse.**
   - One-step gradient descent and momentum fell, and attention and backprop stay at 0.
   - These need chained floating-point arithmetic, which a 1.5B model without tools does poorly.
4. **No forgetting.**
   - DBMS, CN, Spider and MBPP are all within a few items of FT-B (McNemar p ≥ 0.38).
   - FT-C keeps FT-B's significant gains over the base model on DBMS and Spider.
   - 1,200 replay examples in a 2,107-example stage were enough.
5. **Cost:** 70 min of training plus 37 min of generation for the five evaluations (plus model loading) on the laptop iGPU. This is far cheaper than retraining the full mix, which would take about 5.5 h of training plus about 3 h of evaluation.
