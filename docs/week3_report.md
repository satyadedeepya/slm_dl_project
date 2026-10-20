# Week 3 Report: Fine-Tuning and Evaluation

**Week 3 goal (from the plan):** train FT-B (full mix) and FT-A (ablation without the verified generator slice), then evaluate both against the baselines on every task.

> Status: FT-B trained and evaluated; FT-A (ablation) training in progress.

## 1. Training setup

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Method | LoRA (PEFT): r = 16, α = 32, dropout 0.05 on q/k/v/o/gate/up/down projections; **18.46M trainable parameters (1.18%)** |
| Precision | bf16 base weights and LoRA on the Intel Arc iGPU (see 1.1) |
| Loss | assistant answer tokens only; system and user tokens masked (−100) |
| Prompt format | Qwen chat template with the same system prompt as evaluation |
| Optimiser / schedule | AdamW, LR 2e-4, cosine decay, 3% warmup, no weight decay |
| Batch | 1 sequence × 32 gradient-accumulation steps = effective batch 32 |
| Max length | 1,024 tokens (no example is longer) |
| Sampling | length-grouped batches (less padding) |
| Epochs | 1 (see 1.1) |
| Validation | 150 held-out SFT examples every 70 steps and at the end; checkpoint at each evaluation (resumable) |
| Seed | 42 |

**Runs:**

| Run | Training data | Examples | Tokens | Steps |
|---|---|---|---|---|
| **FT-B** | full mix | 8,868 | 2.55M | 278 |
| **FT-A** | full mix minus the verified OS/CN/DBMS generator slice | ~6,590 | ~1.66M | ~206 |

The FT-A vs FT-B comparison measures what the verified generator data adds, with everything else held fixed.

### 1.1 Hardware decisions

Training runs locally on an Intel Core Ultra 5 125H with integrated Arc graphics (8.4 GB of shared memory).

- **4-bit QLoRA was not used.** bitsandbytes 4-bit is not dependable on Intel XPU, so the base model is held in bf16. The LoRA adapter is identical in size and structure, so FT results are comparable to a QLoRA setup.
- **Batch 4 ran out of memory.** With Qwen's 152K-token vocabulary, the output logits alone take ~2.5 GB for 4 × 1,024 tokens. Batch 1 with 32-step accumulation gives the same effective batch.
- **Speed:** about 70–110 s per optimiser step (32 sequences), roughly 90 tokens/s. One FT-B epoch takes about 6 hours. Two epochs per run (the original plan) would take more than 20 hours in total, so both runs use one epoch, which was the user's choice for speed. A free Colab T4 with QLoRA would be several times faster but needs manual setup.

## 2. Evaluation protocol

Each adapter is merged into the base model and evaluated zero-shot with the same harness, prompts and greedy decoding as the baselines B0 (zero-shot) and B1 (3-shot):

| Task | n | Metric |
|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | 400 | accuracy |
| Verified OS / CN / DBMS numericals | 100 each | exact match on the final answer |
| MBPP | 200 | pass@1 (unit tests executed) |
| Spider dev | 200 | execution accuracy |
| GATE CS 2021–2026 test half | 56 | accuracy |

**Reporting:**
- 95% bootstrap confidence intervals for every score.
- Exact McNemar tests against the baselines on identical items.
- Format failures (unparseable answers) listed separately from accuracy.

## 3. Results

### 3.1 Training curves

Validation loss on the 150 held-out SFT examples:

| Step | Epoch | FT-B val loss | FT-A val loss |
|---|---|---|---|
| 70 | 0.25 | 0.733 | |
| 140 | 0.51 | 0.671 | |
| 210 | 0.76 | 0.656 | |
| 278 (end) | 1.00 | **0.650** | |

**FT-B notes:**
- The validation loss fell at every evaluation and was still falling slowly at the end. There is no sign of overfitting after one epoch.
- The training loss swings between about 0.2 and 1.3 within the epoch. This comes from length-grouped sampling: each step holds examples of similar length, and short generator or MCQ answers have much lower per-token loss than long explanations. The mean training loss over the resumed part of the run (steps 71–278) was 0.80.
- **Interruptions:**
  - The run was interrupted twice: a manual sleep at step 101, and a critical-battery hibernate at step 181. The first resumed from `checkpoint-70`. The second paused the process without killing it.
  - The final model is the same as an uninterrupted run, because the data order and optimiser state are restored from the checkpoint.
  - Effective speed was about 62 s per step. Measured from the resume, including the 55-minute pause, the wall time was 4.8 hours.

### 3.2 FT-B against the baselines

Accuracy in % with the 95% bootstrap CI. The p-value is an exact McNemar test of FT-B against B0 on the same items. Full tables and per-family breakdowns are in [`eval/results/RESULTS.md`](../eval/results/RESULTS.md).

| Task | n | B0 zero-shot | B1 3-shot | **FT-B** zero-shot | Δ vs B0 | p (vs B0) |
|---|---|---|---|---|---|---|
| CS-Bench (DSA/OS/CN) | 400 | 51.2 | 58.0 | **55.2** [50.5–60.2] | +4.0 | 0.18 |
| Verified OS numericals | 100 | 4.0 | 4.0 | **5.0** [1–9] | +1.0 | 1.00 |
| Verified CN numericals | 100 | 1.0 | 2.0 | **8.0** [3–13] | +7.0 | **0.039** |
| Verified DBMS FD problems | 100 | 15.0 | 12.0 | **30.0** [21–40] | +15.0 | **0.017** |
| MBPP pass@1 | 200 | 41.5 | 41.5 | **42.5** [36–49] | +1.0 | 0.89 |
| Spider dev (execution) | 200 | 50.5 | – | **60.0** [53–66.5] | +9.5 | **0.011** |
| GATE CS 2021–26 | 56 | 33.9 | – | **33.9** [21–46] | 0.0 | 1.00 |

**Significant gains (p < 0.05):** Spider, the DBMS FD problems and the CN numericals. **No significant change:** CS-Bench, OS numericals, MBPP and GATE. No task got worse.

**Selected breakdowns (B0 → FT-B):**
- DBMS, FD closure and keys: 3.3 → **32.8** (n = 61). This accounts for almost all of the DBMS gain.
- CN: subnetting 0 → 9.1, flow control 0 → 11.8, CRC 0 → 7.7.
- OS: CPU and disk scheduling stay at 0 for all models. Page replacement 3.7 → 7.4.
- CS-Bench: Knowledge 54.7 → 64.3; Reasoning 45.1 → **38.7** (down); OS 39.8 → 50.0.
- Format failures on CS-Bench fell from 21/400 to 7/400. On parsed answers only, CS-Bench is 54.1 → 56.2, so part of the CS-Bench gain is better answer formatting, not new knowledge.

_FT-A results are added when that run finishes._

## 4. Analysis

### 4.1 FT-B (preliminary, before the ablation)

1. **Structured database tasks gain the most.**
   - Spider (+9.5) and FD closure/keys (+29.5 within that family) are both tasks where the answer is a short, checkable artefact (a SQL query, an attribute set) produced by a fixed procedure.
   - The training mix had many examples of exactly these procedures: Spider train, and the DBMS generator.
2. **Long numerical simulations are not learned in one epoch.**
   - On the OS numericals, FT-B copies the training layout perfectly: all 100 answers have Gantt or frame tables and a "Final answer" line.
   - But its state tracking is wrong. For example, it marks frames as unchanged while counting faults, or gives negative waiting times.
   - CPU and disk scheduling stay at 0%. A 1.5B model imitates the shape of a trace without executing it.
   - **Likely fixes:** more epochs; shorter per-step traces; or tool use, where the model writes code that computes the schedule.
3. **General ability is kept.** MBPP (+1.0) and GATE (0.0) did not change, so there is no catastrophic forgetting.
4. **The CS-Bench gain is partly formatting.**
   - Fewer unparseable answers account for about half of the +4.0.
   - The Reasoning subset fell by 6.4 points, a possible small cost of training on short-answer MCQs. It is not significant at this n.
   - 3-shot prompting (B1) is still slightly better than FT-B on CS-Bench (58.0 vs 55.2).
5. **The FT-A ablation is still needed** to attribute the DBMS and CN gains to the verified generator slice and not to the rest of the mix.
