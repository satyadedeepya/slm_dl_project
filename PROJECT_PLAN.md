# CSE-Tutor-1.5B: Fine-Tuning Qwen2.5-1.5B-Instruct for CSE Education

**Type:** One-month course project
**Base model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
**Method:** QLoRA (4-bit) supervised fine-tuning on a mixed, partly *verified* CSE instruction dataset
**Domains:** Data Structures & Algorithms (DSA), Programming, DBMS, Operating Systems (OS), Computer Networks (CN)
**Compute:** One free Colab / Kaggle T4 (16 GB)

---

## 1. Problem Statement & Objectives

Students preparing for CSE exams (university exams, GATE, placement interviews) need an assistant that:
- explains core concepts correctly,
- solves **numerical problems** step by step (CPU scheduling, page faults, subnetting, CRC, normalization),
- writes correct code and SQL,
- runs **locally and cheaply** (no paid API, works on a laptop).

A 1.5B model can run locally, but out of the box it often makes mistakes on CS numerical problems. It computes waiting times wrongly, miscounts page faults and gets subnet ranges wrong, because it pattern-matches instead of following the algorithm.

**The weakness of the common approach:** most student projects ask GPT to write Q&A pairs and fine-tune on them. Those labels are *unverified*, and a small model memorises the teacher's mistakes.

**Our differentiator: verified numerical data.** For OS and CN, we write small **Python generators** that create random problem instances *and compute the correct answer by running the actual algorithm*. Every numerical training label is therefore guaranteed correct. Code and SQL data come from datasets with tests or executable ground truth (MBPP, Spider).

### Objectives
1. Build a ~7K-example CSE instruction dataset covering all 5 subjects, with a verified OS/CN slice.
2. Fine-tune Qwen2.5-1.5B-Instruct with QLoRA on a free T4.
3. Measure the improvement on **held-out** benchmarks (CS-Bench, GATE, Spider, MBPP) against the base model.
4. **Ablation:** does the verified OS/CN slice measurably improve numerical reasoning?
5. Check that general ability is not lost (catastrophic forgetting) using MMLU.
6. Ship a small local chat demo (Gradio).

### Research questions
- **RQ1:** How much does QLoRA fine-tuning improve a 1.5B model on CSE questions compared with zero- and few-shot prompting?
- **RQ2:** Does adding programmatically verified numerical problems improve exact-match accuracy on OS/CN numericals?
- **RQ3:** How much general knowledge (MMLU) is lost after domain fine-tuning?

---

## 2. Literature Review

| # | Paper | Key contribution | Relevance to us |
|---|---|---|---|
| 1 | **LoRA**, Hu et al., 2021 ([arXiv 2106.09685](https://arxiv.org/abs/2106.09685)) | Freeze the base weights; train low-rank update matrices ΔW = BA | Core fine-tuning method |
| 2 | **QLoRA**, Dettmers et al., 2023 ([arXiv 2305.14314](https://arxiv.org/abs/2305.14314)) | 4-bit NF4 base model + LoRA, double quantization, paged optimizers; matches 16-bit fine-tuning quality | Lets us train on a free 16 GB T4 |
| 3 | **Qwen2.5 Technical Report** ([arXiv 2412.15115](https://arxiv.org/abs/2412.15115)) and **Qwen2.5-Coder** ([arXiv 2409.12186](https://arxiv.org/abs/2409.12186)) | Architecture, training data and benchmarks of the base family | Base-model justification; Qwen2.5-Coder-1.5B as a comparison point |
| 4 | **CS-Bench**, Song et al., 2024 ([arXiv 2406.08587](https://arxiv.org/abs/2406.08587)) | First comprehensive CS benchmark: ~10K questions, 26 subfields across DSA, Computer Organization, CN and OS; MCQ, assertion, fill-in and open-ended. Finds CS ability correlates with math and code ability, and reasoning questions are harder than knowledge questions | **Main held-out evaluation set** |
| 5 | **GuideLM**: SFT LLMs as Pedagogical Agents in Programming Education ([arXiv 2502.20527](https://arxiv.org/abs/2502.20527)) | SFT on only **528** student/teacher pairs gave +8% Socratic guidance and +58% economy of words, with a slight accuracy trade-off | Shows that small, high-quality datasets are enough for educational SFT |
| 6 | Fine-Tuning LLMs for Solving DSA Problems Using QLoRA ([IEEE 10969969](https://ieeexplore.ieee.org/document/10969969)) | QLoRA-tuned CodeLlama-7B on LeetCode-style problems | Closest prior work on the DSA slice |
| 7 | **AI-TA**: Intelligent QA Teaching Assistant with Open-Source LLMs ([arXiv 2311.02775](https://arxiv.org/abs/2311.02775)) | Open LLM + RAG + SFT for a CS course forum | Related CS teaching-assistant system |
| 8 | **Small Models, Big Support**: Local LLM framework for educators ([arXiv 2506.05925](https://arxiv.org/abs/2506.05925)) | Small local models for education with RAG/CAG | Motivates a *small, local* model |
| 9 | Large-Scale Real-World Evaluation of an LLM Virtual TA ([arXiv 2506.17363](https://arxiv.org/abs/2506.17363)) | Deployed a VTA to 477 students; survey-based evaluation | Evaluation ideas, human study (future work) |
| 10 | **Self-Instruct**, Wang et al., 2022 ([arXiv 2212.10560](https://arxiv.org/abs/2212.10560)) | Bootstrapping instruction data from seed tasks | Method for the concept Q&A slice |
| 11 | **GLAN**: Synthetic Data (Almost) from Scratch ([arXiv 2402.13064](https://arxiv.org/abs/2402.13064)) | Generate instruction data from a *taxonomy/syllabus* of subjects | We generate concept questions from a CSE syllabus topic list in the same way |
| 12 | Analyzing and Reducing Catastrophic Forgetting in PEFT ([arXiv 2402.18865](https://arxiv.org/abs/2402.18865)) | LoRA still forgets; analysis and mitigation | Justifies the MMLU forgetting check and a small replay mix |
| 13 | **Socratic Debugging Benchmark**, Al-Hossami et al., BEA 2023 ([ACL](https://aclanthology.org/2023.bea-1.57/), [arXiv 2310.03210](https://arxiv.org/abs/2310.03210)) | Multi-turn Socratic tutoring dialogues for buggy code | Future work: tutoring-style evaluation |
| 14 | **Spider**, Yu et al., 2018 ([arXiv 1809.08887](https://arxiv.org/abs/1809.08887)) | Cross-domain text-to-SQL with execution evaluation | DBMS training and evaluation |
| 15 | **MBPP**, Austin et al., 2021 ([arXiv 2108.07732](https://arxiv.org/abs/2108.07732)) | ~1K Python problems with unit tests | Programming training and evaluation |

**Takeaways for our design:**
- QLoRA makes 1.5B fine-tuning cheap; data quality matters more than quantity (GuideLM, LIMA-style findings).
- Small models struggle most on *reasoning-type* CS questions (CS-Bench), so we target numerical and algorithmic problems with verified data.
- Domain SFT can hurt general ability, so we measure it (MMLU) and mix in a small amount of general replay data.
- Public reference point: a LoRA fine-tune of Qwen2.5-1.5B-Instruct on Spider reportedly improved execution accuracy from ~47.5% to ~61.5% on a 200-query subset ([text2sql-lora](https://github.com/synnerg/text2sql-lora)). This is a sanity target for our DBMS slice.

---

## 3. Dataset Survey & Final Choice

### 3.1 Candidates surveyed

| Dataset | Content | License | Decision |
|---|---|---|---|
| [CS-Bench](https://huggingface.co/datasets/CS-Bench/CS-Bench) | ~10K CS questions (DSA, CO, CN, OS), JSONL | CC BY-NC 4.0 | ✅ **Eval only** (never train on it) |
| [GATE CSE Questions (Kaggle)](https://www.kaggle.com/datasets/sakshi2409/gate-cse-question-classification-dataset) | ~2.4K GATE PYQs in 8 subjects (CN, OS, DS, ...) | Check Kaggle card | ✅ **Eval only** (~200 sampled) |
| [Spider](https://yale-lily.github.io/spider) | 10K text-to-SQL pairs, 200 databases | CC BY-SA 4.0 | ✅ Train (train split) + Eval (dev) |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | 974 Python problems with tests | CC BY 4.0 | ✅ Train (train split) + Eval (test) |
| [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | 20K code instructions | See card (research use) | ✅ Train, filtered to DSA/Python (~1.2K) |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | 57 subjects | MIT | ✅ Eval: CS subjects + general subjects for the forgetting check |
| [CS StackExchange](https://huggingface.co/datasets/habedi/stack-exchange-dataset) | Real CS Q&A | CC BY-SA | ⚠️ Optional: noisy, long answers |
| Socratic Debugging Benchmark | Tutoring dialogues | See repo | ⏭️ Future work |
| TACO / APPS | Competitive programming | Apache / MIT | ❌ Too hard and long for 1.5B in 1 month |

### 3.2 Final training mix (~7,000 examples)

| Slice | Source | # Examples | Verified? |
|---|---|---|---|
| DSA + Programming | MBPP train (374) + CodeAlpaca filtered (~1,200) + teacher-written DSA concept Q&A (~400) | ~2,000 | MBPP: tests ✅ |
| DBMS | Spider train (~1,000 sampled) + normalization/keys generator (~200) + concept Q&A (~300) | ~1,500 | SQL executes ✅, generator ✅ |
| OS | **Generators** (~900) + concept Q&A (~600) | ~1,500 | Generator ✅ |
| CN | **Generators** (~900) + concept Q&A (~600) | ~1,500 | Generator ✅ |
| General replay | Alpaca-cleaned / general instructions | ~500 | (prevents forgetting) |

**Split:** 95% train / 5% validation. Evaluation sets are completely separate (Section 6).

---

## 4. Data Pipeline

```
 Syllabus topic list ──► Teacher LLM ──► concept Q&A ──► manual spot-check (10%)
                                                                 │
 OS/CN/DBMS generators ──► problem + step-by-step solution ──────┤  (answers computed = verified)
                                                                 │
 MBPP / CodeAlpaca / Spider ──► filter + reformat ───────────────┤
                                                                 ▼
               normalize ─► dedupe (exact + MinHash) ─► contamination check vs eval sets
                                                                 ▼
                         Qwen chat-template JSONL (messages: system/user/assistant)
```

### 4.1 Generators (the verified slice)

| Subject | Problem types | How the answer is computed |
|---|---|---|
| OS | FCFS, SJF (non-preemptive), SRTF, Round Robin, Priority → avg waiting time / turnaround time, Gantt chart | Discrete-event scheduler simulation |
| OS | FIFO, LRU, Optimal page replacement → page faults | Simulate frames over the reference string |
| OS | Banker's algorithm → safe or unsafe, safe sequence | Safety algorithm |
| OS | Disk scheduling (FCFS, SSTF, SCAN, C-SCAN) → total head movement | Simulation |
| CN | Subnetting/CIDR → network address, broadcast, host range, #hosts | Python `ipaddress` module |
| CN | CRC → remainder / transmitted frame | Polynomial (mod-2) division |
| CN | Transmission + propagation delay, Stop-and-Wait / sliding-window efficiency | Closed-form formulas |
| DBMS | Attribute closure, candidate keys, highest normal form | Closure algorithm over functional dependencies |

Each generator randomizes its parameters (number of processes, burst times, frames, IP prefix, ...) and renders a natural-language question plus a **step-by-step solution template** (explanation, then table, then final answer). Every sample is also re-checked by a unit test.

**Example (FCFS):**

```python
def fcfs(procs):                      # procs: [(name, arrival, burst)]
    t, rows = 0, []
    for name, at, bt in sorted(procs, key=lambda p: p[1]):
        start = max(t, at); t = start + bt
        rows.append((name, at, bt, start, t, t - at, t - at - bt))  # CT, TAT, WT
    return rows
```

> **Q:** Processes P1(arrival 0, burst 5), P2(1, 3), P3(2, 8). Using FCFS, find the average waiting time and turnaround time.
> **A:** Gantt: P1 [0–5] → P2 [5–8] → P3 [8–16].
> TAT = CT − AT: P1 = 5, P2 = 7, P3 = 14 → avg **8.67**.
> WT = TAT − BT: P1 = 0, P2 = 4, P3 = 6 → avg **3.33**.

Because the generators are parameterized, we can also build a **fresh, unseen test set** of 100 problems (different random seed and parameter ranges) for exact-match evaluation.

### 4.2 Concept Q&A (teacher-generated)
- Input: a syllabus topic list (~150 topics, e.g. *"Deadlock: necessary conditions"*, *"TCP 3-way handshake"*, *"B+ tree vs B tree"*, *"Dijkstra vs Bellman-Ford"*).
- A teacher model (free tier, e.g. Qwen2.5-72B-Instruct via HF Inference or Gemini) generates 3–5 questions per topic of mixed types: define, compare, why, example.
- Answers are generated **separately** from questions (GLAN finding: better quality).
- Manually spot-check about 10% and discard bad topics.

### 4.3 Cleaning & contamination control
- Exact + near-duplicate removal (MinHash, Jaccard > 0.8).
- **13-gram overlap check** between training data and CS-Bench / GATE / Spider-dev / MBPP-test; drop any overlapping training sample.
- Length filter: total length ≤ 1024 tokens.

---

## 5. Training Method

**Why QLoRA:** the full 1.5B model in fp16 needs about 3 GB for weights, plus gradients and optimizer states (~12+ GB). That is borderline on a T4. With QLoRA, the base model in 4-bit NF4 takes about 1 GB, and only about 18M LoRA parameters (~1.2%) are trained, so it fits comfortably and is fast.

| Hyperparameter | Value |
|---|---|
| Framework | Unsloth + HF TRL `SFTTrainer` + PEFT |
| Quantization | 4-bit NF4, double quant, compute dtype fp16 (T4 has no bf16) |
| LoRA rank / alpha / dropout | r = 16, α = 32, dropout = 0.05 |
| Target modules | q, k, v, o, gate, up, down projections (all linear) |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Batch | 4 per device × 8 gradient accumulation = effective 32 |
| Epochs | 2 |
| Max sequence length | 1024 |
| Optimizer | paged AdamW 8-bit |
| Loss | Assistant tokens only (mask system/user tokens) |
| Chat format | Qwen2.5 chat template (`<\|im_start\|>` ...) |
| Seed | 42 (repeat the best config with 2 more seeds if time allows) |

**Compute budget:** ~7K examples × ~400 tokens ≈ 2.8M tokens/epoch. On a T4 with Unsloth that is about 25–40 min/epoch, so **under ~1.5 h per run**. Two runs for the ablation fit inside one Colab session each.

**Outputs:** LoRA adapter → merged fp16 model → (optional) GGUF Q4_K_M via llama.cpp for CPU inference.

---

## 6. Evaluation Plan

### 6.1 Models compared

| ID | Model | Purpose |
|---|---|---|
| B0 | Qwen2.5-1.5B-Instruct, zero-shot | Baseline |
| B1 | Qwen2.5-1.5B-Instruct, 3-shot | Stronger prompting baseline (does fine-tuning beat prompt engineering?) |
| FT-A | Fine-tuned **without** the generator slice | Ablation |
| FT-B | Fine-tuned **with** the full mix | Our model |
| Ref | Qwen2.5-7B-Instruct (if time allows, 4-bit) | Upper reference |

### 6.2 Benchmarks & metrics

| Benchmark (held out) | Size | Metric | Measures |
|---|---|---|---|
| CS-Bench EN (DSA, OS, CN; MCQ + assertion) | ~400 | Accuracy, per domain and knowledge-vs-reasoning | Overall CS ability (RQ1) |
| GATE CSE PYQs | ~200 | Accuracy | Exam realism |
| Fresh generator problems (new seed/ranges) | 100 | Exact match on final number | Numerical reasoning (RQ2) |
| Spider dev | 200 | Execution accuracy | DBMS / SQL |
| MBPP test | 200 | pass@1 | Programming |
| MMLU CS subjects (college CS, HS CS, computer security) | ~400 | Accuracy | CS knowledge |
| MMLU general (5 non-CS subjects) | ~500 | Accuracy change | Forgetting (RQ3) |
| Human rating | 30 answers | 1–5 correctness, clarity | Explanation quality |

**Protocol:**
- Greedy decoding (temperature 0) and the same prompt template for every model.
- MCQ answers are extracted by regex (`Answer: X`). Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for MMLU.
- Report **95% bootstrap confidence intervals**, so a +2% difference on 200 questions is not overclaimed.

### 6.3 Expected results (hypotheses, not claims)
- FT-B > B1 > B0 on the generator problems and on CS-Bench reasoning-type questions.
- FT-B > FT-A on numerical problems (validates the verified slice).
- Small MMLU-general drop (< 2 points) thanks to the replay mix.

---

## 7. System Architecture

```
┌────────────────────────── DATA ──────────────────────────┐
│  Generators (OS/CN/DBMS)   Public sets (MBPP/Spider/      │
│  ── verified answers ──    CodeAlpaca)   Teacher LLM Q&A  │
│              └──────────── build_dataset.py ─────────────┘
│                 dedupe · contamination check · chat JSONL │
└───────────────────────────────┬───────────────────────────┘
                                ▼
┌────────────────────────── TRAIN (Colab T4) ──────────────┐
│  Qwen2.5-1.5B-Instruct (4-bit NF4) + LoRA r=16            │
│  Unsloth / TRL SFTTrainer ──► adapter ──► merged / GGUF   │
└───────────────────────────────┬───────────────────────────┘
                                ▼
┌────────────────────────── EVAL ───────────────────────────┐
│  run_eval.py: CS-Bench · GATE · Spider(exec) · MBPP(tests)│
│  · generator exact-match · lm-eval-harness (MMLU)         │
│  ──► results.csv + plots                                  │
└───────────────────────────────┬───────────────────────────┘
                                ▼
┌────────────────────────── DEMO ───────────────────────────┐
│  Gradio chat UI ◄── transformers / llama.cpp (CPU)        │
│  subject selector · streaming answers                     │
└───────────────────────────────────────────────────────────┘
```

### Repository layout
```
finetuning_slm/
├── PROJECT_PLAN.md
├── data/
│   ├── generators/        os_gen.py, cn_gen.py, dbms_gen.py (+ tests)
│   ├── topics.yaml        syllabus topic list for concept Q&A
│   ├── build_dataset.py   download, filter, merge, dedupe, contamination check
│   └── processed/         train.jsonl, val.jsonl
├── train/finetune_qlora.ipynb
├── eval/
│   ├── run_eval.py
│   └── results/
├── demo/app.py
└── report/                 final report + slides
```

---

## 8. Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Training | PyTorch, Hugging Face Transformers, TRL, PEFT, bitsandbytes, **Unsloth** |
| Data | `datasets`, pandas, `datasketch` (MinHash), Python `ipaddress`, `sqlite3` |
| Evaluation | lm-evaluation-harness, custom `run_eval.py`, sqlite3 (Spider execution), sandboxed `subprocess` with timeout (MBPP) |
| Tracking | Weights & Biases (free) or TensorBoard |
| Compute | Google Colab / Kaggle T4 (16 GB) |
| Serving/Demo | Gradio; optionally llama.cpp (GGUF) for CPU |
| Hosting | Hugging Face Hub (adapter + dataset card) |

---

## 9. Four-Week Roadmap

### Week 1: Foundations & baseline
- [ ] Read papers 1–6 and 11–12; write the literature review section of the report.
- [ ] Set up the Colab notebook: load Qwen2.5-1.5B-Instruct in 4-bit, test chat.
- [ ] Download the eval sets (CS-Bench, GATE sample, Spider dev, MBPP test); write `run_eval.py`.
- [ ] **Run baselines B0 and B1**, so the starting numbers are fixed early.
- [ ] Write the OS generators (scheduling, paging, Banker's) + unit tests.

### Week 2: Dataset construction
- [ ] Write the CN and DBMS generators + tests; generate ~2K verified samples.
- [ ] Build the topic list; generate concept Q&A with the teacher LLM; spot-check 10%.
- [ ] Filter and reformat MBPP / CodeAlpaca / Spider.
- [ ] `build_dataset.py`: merge, dedupe, contamination check, chat-template JSONL, split.
- [ ] Write the dataset statistics (counts per subject, token lengths).

### Week 3: Training & evaluation
- [ ] Train **FT-A** (no generator slice) and **FT-B** (full mix).
- [ ] Watch train/val loss; adjust LR/epochs once if it overfits.
- [ ] Run the full evaluation suite on both, plus the MMLU forgetting check.
- [ ] (Optional) Qwen2.5-7B reference numbers.

### Week 4: Analysis, demo, report
- [ ] Error analysis: categorize 50 wrong answers (concept gap / arithmetic slip / format error / hallucination).
- [ ] Plots: per-subject accuracy bars, base vs fine-tuned; loss curves.
- [ ] Gradio demo; optionally a GGUF export for laptop CPU.
- [ ] Final report + slides; push the adapter and dataset card to the HF Hub.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Colab session timeouts / T4 out of memory | Unsloth, seq len 1024, gradient checkpointing, save checkpoints to Drive every 200 steps |
| Teacher LLM produces wrong concept answers | Separate question and answer generation, 10% manual check, prefer verified slices |
| Overfitting to the generator templates | Several phrasing templates per problem type, random parameter ranges, fresh-seed test set |
| Evaluation contamination | 13-gram overlap filter; CS-Bench/GATE never used in training |
| Catastrophic forgetting | 500 general replay samples; low LR; report the MMLU change |
| Small improvement on CS-Bench (knowledge-heavy) | Expected: SFT teaches format and procedure more than new knowledge. Report it honestly and analyse knowledge vs reasoning separately |
| Licence issues (CS-Bench NC) | Academic, non-commercial use only; evaluation only |

---

## 11. Expected Viva / Interview Questions

1. **Why QLoRA instead of full fine-tuning?** Memory math: 4-bit NF4, which LoRA parameters are trained, why quality holds.
2. **What do LoRA rank and alpha mean?** ΔW = BA with rank r; the scale is α/r. What happens with r = 4 vs r = 64?
3. **What is NF4, and why does double quantization save memory?**
4. **Why Qwen2.5-1.5B-Instruct and not the base model?** It already follows instructions and has a chat template, so less data is needed.
5. **Why mask the prompt tokens in the loss?**
6. **How do you know your evaluation isn't contaminated?** n-gram overlap, held-out benchmarks, fresh generator seeds.
7. **What's the difference between your verified data and GPT-generated data?** Labels computed by the algorithm vs predicted by a model.
8. **Does fine-tuning add knowledge or just format?** Discuss knowledge vs reasoning results and when RAG is better.
9. **What is catastrophic forgetting, and how did you measure and mitigate it?**
10. **Why execution accuracy for SQL instead of exact match?**
11. **Is a +3% improvement significant?** Bootstrap confidence intervals, sample size.
12. **Walk through a CPU scheduling / subnetting example your generator produces.** Core CS depth.
13. **How would you deploy this?** Merged weights → GGUF → llama.cpp / vLLM, quantization trade-offs.

---

## 12. Possible Extensions (beyond 1 month)

- **DPO / GRPO with verifier rewards:** use the generators as an automatic reward signal.
- **Tool use:** let the model call the scheduler/subnet calculators at inference time.
- **Backend:** a Java Spring Boot API gateway (SSE streaming, rate limiting, caching, PostgreSQL feedback logs) in front of a vLLM / llama.cpp server.
- **Socratic tutoring mode:** fine-tune on tutoring dialogues and evaluate on the Socratic Debugging Benchmark.
- **More subjects:** TOC, Computer Organization, Compiler Design (generators for DFA minimization, cache hit ratio, etc.).
- **Write-up:** a short paper for a student / education-AI workshop.

---

## 13. References

1. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021. https://arxiv.org/abs/2106.09685
2. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. https://arxiv.org/abs/2305.14314
3. Qwen Team. *Qwen2.5 Technical Report.* 2024. https://arxiv.org/abs/2412.15115
4. Hui et al. *Qwen2.5-Coder Technical Report.* 2024. https://arxiv.org/abs/2409.12186
5. Song et al. *CS-Bench: A Comprehensive Benchmark for LLMs towards Computer Science Mastery.* 2024. https://arxiv.org/abs/2406.08587, https://github.com/csbench/csbench
6. *Supervised Fine-Tuning LLMs to Behave as Pedagogical Agents in Programming Education (GuideLM).* 2025. https://arxiv.org/abs/2502.20527
7. *Fine-Tuning of Large Language Models for Solving DSA Problems Using QLoRA.* IEEE. https://ieeexplore.ieee.org/document/10969969
8. *AI-TA: Towards an Intelligent Question-Answer Teaching Assistant using Open-Source LLMs.* 2023. https://arxiv.org/abs/2311.02775
9. *Small Models, Big Support: A Local LLM Framework for Educator-Centric Content Creation and Assessment.* 2025. https://arxiv.org/abs/2506.05925
10. *A Large-Scale Real-World Evaluation of LLM-Based Virtual Teaching Assistant.* 2025. https://arxiv.org/abs/2506.17363
11. Wang et al. *Self-Instruct.* 2022. https://arxiv.org/abs/2212.10560
12. Li et al. *Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning (GLAN).* 2024. https://arxiv.org/abs/2402.13064
13. *Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning.* 2024. https://arxiv.org/abs/2402.18865
14. Al-Hossami et al. *Socratic Questioning of Novice Debuggers: A Benchmark Dataset and Preliminary Evaluations.* BEA 2023. https://aclanthology.org/2023.bea-1.57/
15. Yu et al. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL.* EMNLP 2018. https://arxiv.org/abs/1809.08887
16. Austin et al. *Program Synthesis with Large Language Models (MBPP).* 2021. https://arxiv.org/abs/2108.07732
17. Hendrycks et al. *Measuring Massive Multitask Language Understanding (MMLU).* ICLR 2021. https://arxiv.org/abs/2009.03300
