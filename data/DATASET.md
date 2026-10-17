# Dataset Card: CSE-Tutor SFT mix

Instruction-tuning data for fine-tuning Qwen2.5-1.5B-Instruct as a CSE tutor (DSA, Programming, DBMS, OS, CN).
Built by `data/build_dataset.py`. Exact counts are in `data/processed/build_report.json`.

> **Status:** current build: 6,566 train / 346 val. Two teacher-generated sources are pending free-tier Gemini quota:
> GATE explanations (`data/build_gate.py --explain`) and textbook-grounded Q&A (`data/build_textbook_qa.py`).
> `data/run_teacher_jobs.cmd` runs them daily and rebuilds the dataset; this card is updated after each rebuild.

## Files

| File | Contents | In git? |
|---|---|---|
| `data/processed/sft_train.jsonl` / `sft_val.jsonl` | training / validation split (5% stratified by source) | yes, except GATE items once added (see below) |
| `data/processed/gen_{os,cn,dbms}_{train,test}.jsonl` | verified generator problems; `_test` files are held-out evaluation sets | yes |
| `data/processed/concepts.jsonl` | teacher concept Q&A | yes |
| `data/processed/textbook_qa.jsonl` | textbook-grounded Q&A (once generated) | yes |
| `data/processed/build_report.json` | per-source counts, drops and token statistics | yes |
| `data/gate_split.json` | GATE question IDs for the train/test split, source URL and commit | yes |
| `data/external/` | downloaded sources: Spider, GATE text, textbooks, caches | **no** (git-ignored) |

Each record has the following fields:
- `id`, `subject`, `topic` and `source`.
- `messages`: a system / user / assistant chat in the Qwen chat format. The system prompt is the same one used in evaluation.
- `n_tokens`: the length with the chat template.

## Sources

| Source | Kept | What it is | Quality gate | License |
|---|---|---|---|---|
| `generator` | 2,400 | OS, CN and DBMS numericals with step-by-step solutions (`data/generators/`) | answers computed by algorithm; unit-tested against textbook examples; every reference solution passes the grader | project code |
| `gate` | pending | GATE CSE previous-year MCQs (OS, CN, DBMS, Algorithms, Programming & DS): all papers ≤ 2020 plus half of 2021–2026, with Gemini explanations | the explanation is written **without seeing the key** and kept only if it reaches the official / GATE Overflow-verified answer | GATE papers © IISc/IITs; text not redistributed in this repo |
| `teacher` | 1,183 | concept Q&A over 108 of 133 syllabus topics (25 pending): open-ended, MCQ and true/false | questions and answers generated in separate calls; MCQ/TF kept only if the answer agrees with the key | generated with Gemini (gemini-3.8-flash 429, gemini-3.5-flash-lite 393, gemini-3-flash-preview 263, gemini-3.6-flash 98) |
| `textbook` | pending | Q&A grounded in passages from open textbooks (below) | answerable from the passage alone; MCQ/TF key agreement | inherits the book's license (CC BY / CC BY-SA) |
| `mbpp` | 311 | MBPP train split | reference code must pass its own unit tests | CC BY 4.0 |
| `spider` | 1,335 | Spider train (text-to-SQL), at most 15 questions per database; prompt = the database's real `CREATE TABLE` schema + question | gold query must execute on its SQLite database | CC BY-SA 4.0 |
| `codealpaca` | 1,184 | `sahil2801/CodeAlpaca-20k`, Python solutions to DSA-style tasks | code must parse (`ast.parse`) | CC BY 4.0 |
| `replay` | 499 | `yahma/alpaca-cleaned`, general instructions | 80–1500 character answers | CC BY 4.0 |

**By subject:** CN 1,152, DBMS 1,978, DSA 330, General 499, OS 1,184, Programming 1,769.

**Change from the previous build:** `b-mc2/sql-create-context` was replaced by Spider train. sql-create-context may contain Spider dev questions, and Spider dev is now an evaluation set. Spider questions are also checked by execution on the real databases.

### Open textbooks used for `textbook`

| Subject | Book | License | Source |
|---|---|---|---|
| CN | Larry Peterson & Bruce Davie, *Computer Networks: A Systems Approach* | CC BY 4.0 | github.com/SystemsApproach/book |
| OS (+ DBMS: transactions chapter) | Max Hailperin, *Operating Systems and Middleware* | CC BY-SA 3.0 | github.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction |
| DSA | Jeff Erickson, *Algorithms* | CC BY 4.0 | jeffe.cs.illinois.edu/teaching/algorithms |
| DSA | Pat Morin, *Open Data Structures (in Python)* | CC BY 2.5 CA | opendatastructures.org |

Copyrighted standard texts (Cormen, Silberschatz/Galvin, Forouzan, Korth) are **not** used. Watt's *Database Design* (CC BY 4.0) was planned for DBMS, but both hosting sites block automated download (a Cloudflare challenge and a token-protected API), so it was left out rather than worked around.

## Cleaning

1. **Exact de-duplication** across all sources, keeping the higher-priority source (generator > gate > teacher > textbook > mbpp > spider > codealpaca > replay). Spider items are compared on database + question, because every prompt on one database shares the whole schema.
2. **Near-duplicate removal** (MinHash LSH on word 5-grams, Jaccard ≥ 0.8) within the non-generator sources.
3. **Decontamination:** an item is dropped if it shares any 13-word n-gram with an evaluation item, or contains a short (6–12 word) evaluation item verbatim. The evaluation items checked are:
   - CS-Bench English test (2,183)
   - MBPP test (500)
   - held-out generator problems (300)
   - Spider dev (1,034)
   - GATE test half (56)

   Generator training problems are checked against the non-generator benchmarks only, because their templates necessarily overlap the generator test set. Their exact questions never overlap.
4. **Length:** at most 1,024 tokens with the Qwen chat template. 162 Spider items were dropped because their schemas are long.

**Dropped:**
- Decontamination: 62 mbpp, 16 codealpaca, 1 replay, 1 spider.
- De-duplication: 1 exact and 1 near-duplicate spider item.
- Length: 162 spider items.

MBPP's train and test splits contain near-duplicate tasks (for example, "find the first repeated character in a given string" appears in both), which accounts for most MBPP drops.

## Statistics

- 6,912 examples, 2,074,695 tokens in total (training split: 1,970,875).
- 300 tokens per example on average, 1,023 maximum.

## Known limitations

- **Teacher answers are not human-verified.** Teacher concept and textbook answers are filtered by key agreement (MCQ/TF) and spot checks only, so open-ended answers can still contain errors. GATE explanations are the most reliable teacher data, because the final answer must match an official or verified key.
- **Small GATE test:** the GATE test half has only 56 questions (DSA 32, CN 10, DBMS 9, OS 5), because only 114 usable MCQs exist from 2021–2026. Its confidence intervals are wide.
- **GATE answer quality:** 763 of the 1,210 GATE answers are community-verified (GATE Overflow) rather than official keys.
- **Generator coverage:** the generators cover a fixed set of problem types and phrasings. Gains on them measure procedural skill on those types, not general reasoning.
- **Licensing:** CS-Bench (CC BY-NC 4.0) is used for evaluation only. CC BY-SA sources (Spider, Hailperin) require derived data to stay CC BY-SA.
