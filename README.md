# CSE-Tutor-1.5B

A QLoRA fine-tune of Qwen2.5-1.5B-Instruct for CSE education (DSA, Programming, DBMS, OS, CN).
The full design is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Status

| Week | Item | Status |
|---|---|---|
| 1 | OS verified generators (scheduling, paging, Banker's, disk) + tests | ✅ done |
| 1 | Evaluation harness (`eval/run_eval.py`) | ✅ done (tested locally with a fake model; real model runs on Colab) |
| 1 | Baselines B0 / B1 | ⏳ run `eval/baseline_colab.ipynb` on a T4 |
| 2 | CN + DBMS generators, concept Q&A, dataset build | — |
| 3 | QLoRA training (FT-A, FT-B) + evaluation | — |
| 4 | Error analysis, demo, report | — |

## Layout

```
data/generators/common.py    Problem record, formatting, number parsing
data/generators/os_gen.py    OS problem generators + solvers (CLI)
data/generators/grading.py   grades model output against a problem's `final` spec
data/processed/              generated JSONL (gen_os_train: 900, gen_os_test: 100)
eval/run_eval.py             benchmark runner (csbench, gen_os, mbpp, spider, gate)
eval/baseline_colab.ipynb    Week-1 baseline notebook
tests/test_os_gen.py         textbook (Silberschatz) cases + property tests
```

## Commands

```bash
# Tests (standard library only)
python -m unittest discover -s tests -t . -v

# Generate OS problems
python -m data.generators.os_gen --n 900 --seed 42 --out data/processed/gen_os_train.jsonl
python -m data.generators.os_gen --n 100 --seed 2026 --split test --out data/processed/gen_os_test.jsonl

# Evaluate (GPU)
python eval/run_eval.py --run-name B0_zeroshot
python eval/run_eval.py --run-name B1_3shot --fewshot 3
```

Results go to `eval/results/<run>/` (one JSONL per task with every prompt, output and score, plus
`summary.json`) and are appended to `eval/results/summary.csv`, with 95% bootstrap confidence intervals.

## Conventions the generators state in every question

Textbooks differ on tie-breaking, so each question states its convention, and the evaluation sticks to it:
- **Scheduling ties:** earlier arrival first, then the lower process number.
- **Round Robin:** a process that arrives during a time slice joins the queue *before* the preempted process is put back.
- **Priority:** a smaller number means a higher priority.
- **Disk scheduling:** SCAN goes all the way to the end of the disk; C-SCAN and C-LOOK count the return seek.
- **Banker's algorithm:** any valid safe sequence is accepted. The grader re-runs the safety check on the sequence the model gives.

## Evaluation data notes
- **CS-Bench** (CC BY-NC 4.0) is used only for evaluation. The English test split is filtered to DSA, OS and CN multiple-choice and true/false questions, and a stratified sample of 400 is taken. Few-shot examples come from its separate validation split.
- **GATE:** no public dataset with answer keys was found. The Kaggle GATE set is for topic classification. To include GATE, build a JSONL of about 200 past-paper MCQs (`question, A, B, C, D, answer, subject`) and pass it with `--gate-file`.
- **Spider:** download it from https://yale-lily.github.io/spider and pass the folder with `--spider-dir`.
