# CSE-Tutor-1.5B

A QLoRA fine-tune of Qwen2.5-1.5B-Instruct for CSE education (DSA, Programming, DBMS, OS, CN).
The full design is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Status

| Week | Item | Status |
|---|---|---|
| 1 | OS verified generators (scheduling, paging, Banker's, disk) + tests | ✅ done |
| 1 | Evaluation harness (`eval/run_eval.py`) | ✅ done (runs on NVIDIA CUDA, Intel Arc XPU or CPU) |
| 1 | Baselines B0 (zero-shot) / B1 (3-shot) | ✅ done, see [Week 1 report](docs/week1_report.md) and [results](eval/results/RESULTS.md) |
| 2 | CN + DBMS generators, concept Q&A, dataset build | ✅ built, see [Week 2 report](docs/week2_report.md) and [dataset card](data/DATASET.md). Spider, GATE PYQs and open textbooks added; GATE explanations, textbook Q&A and 25 concept topics pending Gemini quota (`data/run_teacher_jobs.cmd`) |
| 3 | QLoRA training (FT-A, FT-B) + evaluation | — |
| 4 | Error analysis, demo, report | — |

## Layout

```
data/generators/common.py    Problem record, shared driver/CLI, formatting, number parsing
data/generators/os_gen.py    OS generators: scheduling, paging, Banker's, disk
data/generators/cn_gen.py    CN generators: subnetting, CRC, checksum, delays, window utilisation
data/generators/dbms_gen.py  DBMS generators: closure, candidate keys, normal forms, lossless join
data/generators/grading.py   grades model output against a problem's `final` spec
data/concepts/topics.json    133 syllabus topics for teacher concept Q&A
data/build_concepts.py       Gemini concept Q&A (batched, resumable, key read from .env)
data/build_gate.py           GATE PYQs: download, clean, year split, verified explanations
data/build_textbook_qa.py    Q&A grounded in open-licensed textbooks
data/run_teacher_jobs.cmd    daily runner for all Gemini jobs + dataset rebuild
data/build_dataset.py        merge sources, dedupe, decontaminate, split -> sft_train/sft_val
data/DATASET.md              dataset card: sources, licenses, filters, counts
data/processed/              gen_*_{train,test}.jsonl, concepts.jsonl, sft_{train,val}.jsonl, build_report.json
eval/run_eval.py             benchmark runner (csbench, gen_os, gen_cn, gen_dbms, mbpp, spider, gate; --rescore)
data/external/               (git-ignored) Spider, GATE text, textbooks, caches
eval/summarize.py            builds eval/results/RESULTS.md (CIs, McNemar tests, breakdowns)
eval/run_baselines.cmd       runs B0 and B1 on every task (Windows)
eval/baseline_colab.ipynb    Week-1 baseline notebook (Colab alternative)
eval/results/                per-run outputs, summary.csv, RESULTS.md
docs/week{1,2}_report.md     weekly reports
tests/                       textbook cases, property tests, grader tests (36)
```

## Commands

```bash
# Tests (standard library only)
python -m unittest discover -s tests -t . -v

# Build the training data (verified generators, teacher Q&A, public datasets)
python -m data.generators.cn_gen --n 900 --seed 42 --out data/processed/gen_cn_train.jsonl
python -m data.generators.dbms_gen --n 600 --seed 42 --out data/processed/gen_dbms_train.jsonl
python data/build_concepts.py        # needs GEMINI_API_KEY in .env; resumable
python data/build_dataset.py         # -> data/processed/sft_train.jsonl, sft_val.jsonl

# Generate OS problems
python -m data.generators.os_gen --n 900 --seed 42 --out data/processed/gen_os_train.jsonl
python -m data.generators.os_gen --n 100 --seed 2026 --split test --out data/processed/gen_os_test.jsonl

# Evaluate (GPU)
python eval/run_eval.py --run-name B0_zeroshot
python eval/run_eval.py --run-name B1_3shot --fewshot 3

# Re-score saved outputs (no GPU), then rebuild the comparison table
python eval/run_eval.py --run-name B0_zeroshot --rescore
python eval/summarize.py
```

### Local setup on an Intel Arc GPU (Windows)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
.venv/Scripts/python -m pip install transformers accelerate peft datasets numpy
# Hugging Face's Xet transfer can stall on Windows; plain HTTP downloads reliably.
HF_HUB_DISABLE_XET=1 .venv/Scripts/python eval/run_eval.py --run-name B0_zeroshot
```

Measured on a Core Ultra 5 125H (Arc iGPU, 8.4 GB shared memory, fp16):
- Each decoding step costs about 0.35–0.45 s whatever the batch size, because the time goes into GPU kernel launch overhead rather than compute.
- Throughput therefore scales with batch size: 2.8 tok/s at batch 1, 24 at batch 8, 79 at batch 32 and **144 at batch 64** (3.5 GB peak).
- The default `--batch-size` is 64 for this reason.
- The first call with a new input shape adds about 10 s of one-time kernel compilation.

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
