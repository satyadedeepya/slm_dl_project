# Week 3 Report: Fine-Tuning and Evaluation

**Week 3 goal (from the plan):** train FT-B (full mix) and FT-A (ablation without the verified generator slice), then evaluate both against the baselines on every task.

> Status: FT-B trained; FT-B evaluation in progress. Results sections are filled in as runs finish.

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

_Evaluation results for FT-B and FT-A against B0 and B1 (McNemar p-values, per-family breakdowns, format failures) are added as the runs finish._

## 4. Analysis

_To be filled in._
