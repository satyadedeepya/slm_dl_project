# Week 1 Report: Foundations and Baselines

**Project:** CSE-Tutor-1.5B, a QLoRA fine-tune of Qwen2.5-1.5B-Instruct for CSE education
**Week 1 goal (from the plan):** literature review, environment setup, baseline evaluation of the untuned model, and the first verified data generators.

## 1. Summary

| Deliverable | Status |
|---|---|
| Literature review and dataset survey (`PROJECT_PLAN.md`) | ✅ Done |
| Verified OS problem generators + grader (`data/generators/`) | ✅ Done, 21 unit tests passing |
| Generated OS data: 900 train (seed 42) / 100 held-out test (seed 2026) | ✅ Done, 0 overlap |
| Evaluation harness (`eval/run_eval.py`) | ✅ Done (CUDA / Intel XPU / CPU) |
| Baseline B0 (zero-shot) | ✅ Done |
| Baseline B1 (3-shot) | ✅ Done |

**Main finding:** the untuned 1.5B model **cannot do OS numericals at all**: 4% exact match zero-shot and still 4% with 3 worked examples in the prompt. It is also weakest on OS conceptual questions. This is the gap Weeks 2–3 target with verified training data.

## 2. Verified OS problem generators

Four problem families. Every answer is computed by running the algorithm, never by an LLM:

| Family | Variants | Solver |
|---|---|---|
| CPU scheduling | FCFS, SJF, SRTF, Round Robin, Priority (non-preemptive and preemptive) | discrete-event simulation (Gantt chart, CT/TAT/WT/RT) |
| Page replacement | FIFO, LRU, Optimal | frame simulation with a step-by-step trace |
| Deadlock avoidance | Banker's safety algorithm (safe/unsafe; unsafe cases get partway) | pass-based safety algorithm |
| Disk scheduling | FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK | head-path simulation |

**Correctness checks** (`tests/test_os_gen.py`):
- **Silberschatz worked examples:** FCFS WT = 17, SJF = 7, SRTF = 6.5, RR(q=4) = 5.67, Priority = 8.2; FIFO/LRU/OPT faults = 15/12/9; Belady's anomaly 9 → 10; Banker's safe sequence ⟨P1, P3, P4, P0, P2⟩; disk FCFS 640, SSTF 236, C-SCAN 382.
- **Property tests:** Optimal never has more faults than FIFO or LRU, and every disk algorithm serves every request.
- **Self-consistency:** 6,000 generated reference solutions all pass the grader.

**Ambiguity control:** textbooks differ on tie-breaking, Round Robin queueing order and whether C-SCAN counts the return seek, so each question states its convention.

**Grading:** the grader parses the `Final answer` line and requires the expected numbers in order, with a tolerance of 0.011. For Banker's it accepts any valid safe sequence by re-running the safety check on the model's sequence.

## 3. Evaluation setup

| Benchmark | Items | Metric | Notes |
|---|---|---|---|
| CS-Bench (English test) | 400 | accuracy | DSA/OS/CN, MCQ + true/false, stratified sample (seed 0); few-shot examples from the separate validation split |
| Verified OS numericals | 100 | exact match | held-out generator problems (seed 2026, harder parameter ranges) |
| MBPP (full test split) | 200 | pass@1 | generated code executed against the unit tests (10 s timeout) |

- **Decoding:** greedy decoding, the same system prompt for all runs, and Qwen's chat template.
- **Confidence intervals:** 95% bootstrap, 1,000 resamples.
- **Paired comparison:** exact McNemar test on identical items.
- **Hardware:** everything ran locally on an **Intel Core Ultra 5 125H with integrated Arc graphics** (8.4 GB shared memory) using PyTorch XPU, in fp16.

## 4. Results

| Task | n | B0 zero-shot | B1 3-shot | McNemar p (B0 vs B1) |
|---|---|---|---|---|
| CS-Bench (DSA/OS/CN) | 400 | **51.2** [46.8–56.0] | **58.0** [53.0–63.0] | 0.013 |
| Verified OS numericals | 100 | **4.0** [1.0–8.0] | **4.0** [1.0–8.0] | 1.000 |
| MBPP pass@1 | 200 | **41.5** [34.5–48.0] | **41.5** [35.0–48.0] | 1.000 |

### CS-Bench breakdown (accuracy %)

| Group | n | B0 | B1 | Δ |
|---|---|---|---|---|
| DSA | 144 | 56.9 | 57.6 | +0.7 |
| CN | 138 | 55.1 | 61.6 | +6.5 |
| **OS** | 118 | **39.8** | 54.2 | +14.4 |
| Knowledge | 258 | 54.7 | 63.6 | +8.9 |
| **Reasoning** | 142 | **45.1** | 47.9 | **+2.8** |
| Multiple-choice | 300 | 51.3 | 57.3 | +6.0 |
| True/False | 100 | 51.0 | 60.0 | +9.0 |

### Verified OS numericals by family (B0 → B1, exact match %)

| Family | n | B0 | B1 |
|---|---|---|---|
| CPU scheduling | 43 | 0.0 | 0.0 |
| Page replacement | 27 | 3.7 | 0.0 |
| Banker's algorithm | 12 | 25.0 | 33.3 |
| Disk scheduling | 18 | 0.0 | 0.0 |

The Banker's numbers overstate ability. B1 called all 3 unsafe states unsafe but gave a valid safe sequence for only 1 of 9 safe states, so most of its credit comes from a bias toward answering "unsafe". B0 found valid sequences for 2 of 9.

## 5. Analysis

1. **OS is the weak spot.** Zero-shot, OS trails DSA and CN by about 16 points on CS-Bench (39.8 vs 55–57), and the model essentially cannot do OS numericals: **0% on CPU and disk scheduling**. Its 4 correct answers were 3 Banker's verdicts and 1 FIFO count. Inspected failures show the model knows the *vocabulary* (it lays out Gantt charts and frame tables) but executes the algorithm wrongly: wrong queue order, miscounted faults, an "unsafe" state declared safe.
2. **Few-shot prompting helps knowledge, not reasoning.** 3-shot gains +8.9 points on knowledge questions but only +2.8 on reasoning questions. Worked examples in the prompt teach *what an answer looks like*, not *how to run an algorithm*. That motivates fine-tuning on verified step-by-step solutions instead of relying on prompting.
3. **Part of the 3-shot gain is answer format.** Unparseable CS-Bench answers fell from 21/400 (5.2%) to 4/400, and average answer length dropped from 897 to 87 characters, because the examples taught the model to answer directly. On parseable answers only, the gain is 54.1 → 58.6. When we evaluate fine-tuned models in Week 3, we will report format-failure rate separately so format learning is not mistaken for knowledge gain.
4. **The gain is statistically significant but modest.** On CS-Bench, B1 fixed 69 items and broke 42 (McNemar p = 0.013).
5. **Prompting does not move procedural skills.** On the OS numericals and MBPP, 3-shot is identical to zero-shot (4.0% and 41.5%; McNemar p = 1.0 for both, with 3 vs 3 and 22 vs 22 discordant items). On MBPP the examples did change the *form* of answers: missing code blocks fell from 4 to 0 and answers got about 45% shorter. They did not make the code more correct. Together with point 2, this is the clearest evidence that the procedural gap needs training data rather than prompt engineering.

## 6. Engineering notes

- **Throughput on the Arc iGPU is set by a fixed cost per decoding step** (about 0.35–0.45 s), not by compute. Batching is therefore critical: 2.8 tok/s at batch 1, 144 tok/s at batch 64. We used batch 32 for B0 and 16/8 for B1 because of memory limits.
- **Memory:** the GPU shares the laptop's 15.4 GB of RAM. Three fixes followed from this week's problems:
  - Weights now load directly onto the GPU, avoiding a second copy in RAM.
  - The 3-shot OS prompts (three full worked solutions) ran out of memory at batch 16, so that task ran at batch 8.
  - The harness now saves results after every task and can `--rescore` saved outputs without the model.
- **Downloads:** Hugging Face's Xet transfer stalled repeatedly on Windows, and `HF_HUB_DISABLE_XET=1` fixed it.

## 7. Implications for Weeks 2–3

- **Prioritise reasoning/procedural data**, especially the verified generator slice: CPU scheduling and disk scheduling are at 0%.
- **Add the CN and DBMS generators next**: subnetting, CRC, delays, closure and keys. CS-Bench shows CN is easier for the model than OS but still only 55–62%.
- **Comparisons for Week 3:** FT-A and FT-B must beat **B1** (the stronger baseline), not just B0. Report the format-failure rate and McNemar p-values for every comparison.
- **Train with the same chat template and system prompt** used in evaluation, so gains are not confounded by prompt mismatch.
