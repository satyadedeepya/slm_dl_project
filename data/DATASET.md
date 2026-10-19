# Dataset Card: CSE-Tutor SFT mix

Instruction-tuning data for fine-tuning Qwen2.5-1.5B-Instruct as a CSE tutor (DSA, Programming, DBMS, OS, CN).
Built by `data/build_dataset.py`. Exact counts are in `data/processed/build_report.json`.

**Final Week 2 build: 9,335 examples (8,868 train / 467 val), 2.68M tokens.**

## Files

| File | Contents | In git? |
|---|---|---|
| `data/external/sft/sft_{train,val}_full.jsonl` | **training input**: all sources | no (contains exam-paper text) |
| `data/processed/sft_{train,val}.jsonl` | same splits **without** the exam-PYQ items (8,357 examples) | yes |
| `data/processed/gen_{os,cn,dbms}_{train,test}.jsonl` | verified generator problems; `_test` files are held-out evaluation sets | yes |
| `data/processed/concepts.jsonl` | teacher concept Q&A | yes |
| `data/processed/textbook_qa.jsonl` | textbook-grounded Q&A | yes |
| `data/processed/build_report.json` | per-source counts, drops and token statistics | yes |
| `data/gate_split.json` | exam-PYQ question IDs for the train/test split, source URL and commit | yes |
| `data/external/` | downloaded sources: Spider, exam papers, textbooks, caches | no (git-ignored) |

To rebuild everything:
1. `python data/build_gate.py`
2. `python data/build_textbook_qa.py --prepare-only`
3. `data/run_teacher_jobs.cmd`

All Gemini outputs are cached, so a rebuild that finds the caches present makes no new API calls.

Each record has the following fields:
- `id`, `subject`, `topic` and `source`.
- `messages`: a system / user / assistant chat in the Qwen chat format. The system prompt is the same one used in evaluation.
- `n_tokens`: the length with the chat template.

## Sources

| Source | Kept | What it is | Quality gate | License |
|---|---|---|---|---|
| `generator` | 2,400 | OS, CN and DBMS numericals with step-by-step solutions (`data/generators/`) | answers computed by algorithm; unit-tested against textbook examples; every reference solution passes the grader | project code |
| `gate` | 978 | previous-year exam MCQs (OS, CN, DBMS, Algorithms, Programming & DS) with worked explanations: GATE CS (≤ 2020 plus half of 2021–2026), GATE IT/DA and ISRO CSE papers | Gemini solves each question **without seeing the key**; kept only if its answer equals the official or GATE Overflow-verified key (990 of 1,154 agreed; 144 disagreed and were dropped; 20 got no usable response) | exam papers © IISc/IITs/ISRO; text not redistributed in this repo |
| `teacher` | 1,458 | concept Q&A over all 133 syllabus topics (`data/concepts/topics.json`): open-ended, MCQ and true/false | questions and answers generated in separate calls; MCQ/TF kept only if the answer agrees with the key (5 rejected) | Gemini-generated |
| `textbook` | 1,170 | Q&A grounded in passages from four open textbooks (below): 705 open-ended, 235 MCQ, 230 true/false | answerable from the passage alone; MCQ/TF key agreement | inherits the book's license (CC BY / CC BY-SA) |
| `mbpp` | 311 | MBPP train split | reference code must pass its own unit tests | CC BY 4.0 |
| `spider` | 1,335 | Spider train (text-to-SQL), at most 15 per database; prompt = real `CREATE TABLE` schema + question | gold query must execute on its SQLite database | CC BY-SA 4.0 |
| `codealpaca` | 1,184 | `sahil2801/CodeAlpaca-20k`, Python solutions to DSA-style tasks | code must parse (`ast.parse`) | CC BY 4.0 |
| `replay` | 499 | `yahma/alpaca-cleaned`, general instructions | 80–1500 character answers | CC BY 4.0 |

**By subject:** CN 1,724, DBMS 2,451, DSA 1,277, General 499, OS 1,615, Programming 1,769.

**Teacher models** (Gemini free tier, which allows 20 requests per day per model, so work is spread across models; every item records its model):

| Source | Items per model |
|---|---|
| exam PYQs | gemini-3.1-flash-lite 649, gemini-3.8-flash 247, gemini-3-flash-preview 66, gemini-3.5-flash-lite 28 |
| concepts | gemini-3.5-flash-lite 470, gemini-3.8-flash 462, gemini-3-flash-preview 395, gemini-3.6-flash 131 |
| textbook | gemini-3.8-flash 965, gemini-3.6-flash 115, gemini-3-flash-preview 90 |

### Open textbooks used for `textbook`

| Subject | Book | Items | License | Source |
|---|---|---|---|---|
| CN | Larry Peterson & Bruce Davie, *Computer Networks: A Systems Approach* | 300 | CC BY 4.0 | github.com/SystemsApproach/book |
| OS (+ DBMS: transactions chapter) | Max Hailperin, *Operating Systems and Middleware* | 290 | CC BY-SA 3.0 | github.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction |
| DSA | Jeff Erickson, *Algorithms* | 295 | CC BY 4.0 | jeffe.cs.illinois.edu/teaching/algorithms |
| DSA | Pat Morin, *Open Data Structures (in Python)* | 285 | CC BY 2.5 CA | opendatastructures.org |

**Excluded books:**
- **Copyrighted standard texts** (Cormen, Silberschatz/Galvin, Forouzan, Korth) are not used.
- **Watt, *Database Design*** (CC BY 4.0) was planned for DBMS. Both hosting sites block automated download (a Cloudflare challenge and a token-protected API), so it was left out rather than worked around.

## Cleaning

1. **Exact de-duplication** across all sources, keeping the higher-priority source (generator > gate > teacher > textbook > mbpp > spider > codealpaca > replay). Spider items are compared on database + question, because every prompt on one database shares the whole schema.
2. **Near-duplicate removal** (MinHash LSH on word 5-grams, Jaccard ≥ 0.8) within the non-generator sources.
3. **Decontamination:** an item is dropped if it shares any 13-word n-gram with an evaluation item, or contains a short (6–12 word) evaluation item verbatim. The evaluation items checked are:
   - CS-Bench English test (2,183)
   - MBPP test (500)
   - held-out generator problems (300)
   - Spider dev (1,034)
   - GATE test half (56)

   Generator problems are checked against the non-generator benchmarks only, because their templates necessarily overlap the generator test set. Their exact questions never overlap.
4. **Length:** at most 1,024 tokens with the Qwen chat template.

**Dropped:**

| Reason | Items |
|---|---|
| Decontamination | 62 mbpp, 16 codealpaca, 3 exam PYQs, 1 replay, 1 spider |
| De-duplication | 8 exam PYQs (repeated questions across papers), 2 spider |
| Too long | 162 spider (long schemas), 1 exam PYQ |

MBPP's train and test splits contain near-duplicate tasks (for example, "find the first repeated character in a given string" appears in both), which accounts for most MBPP drops.

## Statistics

- 9,335 examples, 2,677,157 tokens in total (training split: 2,546,233).
- 287 tokens per example on average, 1,023 maximum.

## Known limitations

- **Open-ended teacher answers are not human-verified.** They are covered only by spot checks, while exam explanations are anchored to official or verified keys and MCQ/TF items to their keys.
- **Mixed teacher strength:** about two-thirds of the exam explanations come from Flash-Lite models. Their answer-key agreement was lower (about 85%, versus about 95% for Flash), but disagreeing explanations were dropped.
- **Small GATE test:** the GATE test half has only 56 GATE CS questions (DSA 32, CN 10, DBMS 9, OS 5), so its confidence intervals are wide.
- **GATE answer quality:** 763 of the 1,210 exam answers are community-verified (GATE Overflow) rather than official keys.
- **Generator coverage:** the generators cover a fixed set of problem types and phrasings. Gains on them measure procedural skill on those types, not general reasoning.
- **Licensing:** CS-Bench (CC BY-NC 4.0) is used for evaluation only. CC BY-SA sources (Spider, Hailperin) require derived data to stay CC BY-SA.
