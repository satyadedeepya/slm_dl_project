# CSE-Tutor-1.5B

A LoRA fine-tune of Qwen2.5-1.5B-Instruct for CSE education (DSA, Programming, DBMS, OS, CN and Deep Learning), trained on a mix that includes 3,000 programmatically verified OS/CN/DBMS/DL problems.

- **Research paper:** [paper/CSE-Tutor_paper.pdf](paper/CSE-Tutor_paper.pdf) (source: `paper/paper.html`)
- **Final report:** [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md); original design: [PROJECT_PLAN.md](PROJECT_PLAN.md)
- **Chat with the model:** `python demo/server.py`, then open http://127.0.0.1:8000

## Quick start

```bash
python demo/server.py          # chat interface: conversations, model switch (FT-E / FT-D / FT-C / FT-B / FT-A / base), streaming, plain-text math, code
python demo/app.py             # Gradio: verified practice problems graded automatically
paper/build_pdf.cmd            # rebuild the paper PDF with headless Edge (Windows)
```

The chat server needs the trained adapters in `outputs/<run>/adapter` (not in git; produced by `train/train_lora.py`).

## Status

| Week | Item | Status |
|---|---|---|
| 1 | OS verified generators (scheduling, paging, Banker's, disk) + tests | ✅ done |
| 1 | Evaluation harness (`eval/run_eval.py`) | ✅ done (runs on NVIDIA CUDA, Intel Arc XPU or CPU) |
| 1 | Baselines B0 (zero-shot) / B1 (3-shot) | ✅ done, see [Week 1 report](docs/week1_report.md) and [results](eval/results/RESULTS.md) |
| 2 | CN + DBMS generators, concept Q&A, GATE/ISRO PYQs, Spider, open-textbook Q&A, dataset build | ✅ done: 9,335 examples (8,868 train / 467 val). See [Week 2 report](docs/week2_report.md) and [dataset card](data/DATASET.md) |
| 3 | LoRA training (FT-A ablation, FT-B full mix) + evaluation | ✅ done. FT-B vs base: Spider 50.5 → 60.0, DBMS FD 15 → 30, CN numericals 1 → 8 (all p < 0.05). The FT-A ablation ties these gains to the verified generator data. See [Week 3 report](docs/week3_report.md) |
| 4 | Error analysis, figures, demo, final report | ✅ done. 50 labelled errors (52% procedure / state tracking), Gradio demo (`demo/app.py`). See [Week 4 report](docs/week4_report.md) and the **[final report](docs/FINAL_REPORT.md)** |

| + | Explanation quality (FT-D, FT-E): blind pairwise judge vs the base model, teaching-style data, plain-text math in the chat app | ✅ done. See the [explanation stage report](docs/explanation_stage_report.md) |
| + | Deep Learning stage (FT-C), chat interface, research paper | ✅ done. Verified DL generators (10 problem types, checked against PyTorch) + DL concept Q&A, continued training from FT-B. See the [DL stage report](docs/dl_stage_report.md) |

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
train/train_lora.py          LoRA SFT (assistant-only loss, resumable checkpoints, --exclude-source for ablations)
train/run_week3.cmd          Week 3 pipeline: train + evaluate FT-B, then FT-A (Windows)
train/week3_progress.py      progress meter read from outputs/*.log
eval/plots.py                accuracy and loss figures -> docs/figures/
eval/error_sample.py         stratified sample of wrong answers for error analysis
eval/error_report.py         tallies the hand labels in eval/error_analysis/
data/generators/dl_gen.py    DL generators: conv/pool shapes, parameter counts, forward pass, softmax-CE, GD, backprop, momentum, attention, metrics
data/build_dl_stage.py       DL continued-training set (DL generators + DL concepts + replay)
train/run_ftc.cmd            FT-C pipeline: DL stage training + evaluation
demo/server.py               chat server (FastAPI, streaming, named LoRA adapters) + demo/static/index.html
paper/                       research paper (HTML source, PDF, build script)
demo/app.py                  Gradio demo: chat + verified practice problems, adapter on/off
docs/week{1,2,3,4}_report.md weekly reports
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

# Fine-tune (adapter -> outputs/<run>/adapter) and evaluate the merged model
python train/train_lora.py --run-name ft-b --epochs 1 --batch-size 1 --grad-accum 32
python train/train_lora.py --run-name ft-a --exclude-source generator --epochs 1 --batch-size 1 --grad-accum 32
python eval/run_eval.py --run-name FT-B --adapter outputs/ft-b/adapter --batch-size 32

# Demo at http://127.0.0.1:7860 (needs outputs/ft-b/adapter from the training step)
python demo/app.py

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
- **GATE:** `data/build_gate.py` builds the PYQ set from a pinned public archive. Papers up to 2020 are used for training; 2021–2026 are split between training and a held-out GATE CS test set (n = 56). The exam text is git-ignored and never pushed; only scores are committed.
- **Spider:** download it from https://yale-lily.github.io/spider and pass the folder with `--spider-dir`.
