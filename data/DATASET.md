# Dataset Card: CSE-Tutor SFT mix

Instruction-tuning data for fine-tuning Qwen2.5-1.5B-Instruct as a CSE tutor (DSA, Programming, DBMS, OS, CN).
Built by `data/build_dataset.py`. Exact counts are in `data/processed/build_report.json`.

## Files

| File | Contents |
|---|---|
| `data/processed/sft_train.jsonl` | training split (6,241 examples) |
| `data/processed/sft_val.jsonl` | validation split (329 examples, 5% stratified by source) |
| `data/processed/gen_{os,cn,dbms}_train.jsonl` | verified generator problems (inputs to the build) |
| `data/processed/gen_{os,cn,dbms}_test.jsonl` | **held-out** generator problems, evaluation only |
| `data/processed/concepts.jsonl` | teacher concept Q&A (input to the build) |
| `data/processed/build_report.json` | per-source counts, drops and token statistics |

Each record has the following fields:
- `id`, `subject`, `topic` and `source`.
- `messages`: a system / user / assistant chat in the Qwen chat format. The system prompt is the same one used in evaluation.
- `n_tokens`: the length with the chat template.

## Sources

| Source | Kept | What it is | Quality gate | License |
|---|---|---|---|---|
| `generator` | 2,400 | OS, CN and DBMS numericals with step-by-step solutions (`data/generators/`) | answers computed by algorithm; unit-tested against textbook examples; every reference solution passes the grader | project code (this repo) |
| `teacher` | 1,183 | concept Q&A over 108 of the 133 syllabus topics in `data/concepts/topics.json` (the remaining 25 (22 DBMS, 3 CN TCP topics) are pending free-tier quota): open-ended, MCQ and true/false | questions and answers generated in separate calls; MCQ/TF kept only if the answer agrees with the key | generated with Gemini Flash-family models (items per model: gemini-3.8-flash (429), gemini-3.5-flash-lite (393), gemini-3-flash-preview (263), gemini-3.6-flash (98)) (Google Gemini API terms) |
| `mbpp` | 311 | MBPP train split | reference code must pass its own unit tests | CC BY 4.0 |
| `sql` | 1,000 | `b-mc2/sql-create-context` (Spider + WikiSQL questions with CREATE TABLE context); only queries with JOIN / GROUP BY / ORDER BY / HAVING / set operations / subqueries | query must be accepted by SQLite against the given schema | CC BY 4.0 |
| `codealpaca` | 1,176 | `sahil2801/CodeAlpaca-20k`, Python solutions to DSA-style tasks | code must parse (`ast.parse`) | CC BY 4.0 |
| `replay` | 500 | `yahma/alpaca-cleaned`, general instructions | 80–1500 character answers | CC BY 4.0 |

**By subject:** CN 1,152, DBMS 1,643, DSA 330, General 500, OS 1,184, Programming 1,761

## Cleaning

1. **Exact de-duplication** of normalised user prompts across all sources, keeping the higher-priority source (generator > teacher > mbpp > sql > codealpaca > replay).
2. **Near-duplicate removal** (MinHash LSH on word 5-grams, Jaccard ≥ 0.8) within the non-generator sources. Generator problems share templates by design and differ in their numbers.
3. **Decontamination:** an item is dropped if it shares any 13-word n-gram with an evaluation item, or contains a short (6–12 word) CS-Bench item verbatim. The evaluation items checked are:
   - CS-Bench English test (all 2,183 items)
   - MBPP test (500)
   - held-out generator problems (300)

   Generator training problems are checked against CS-Bench and MBPP only, because their templates necessarily overlap the generator test set. Their exact questions never overlap: the test split uses different seeds and parameter ranges, and that is verified.
4. **Length:** at most 1,024 tokens with the Qwen chat template.

**Dropped:** 24 codealpaca contamination, 62 mbpp contamination. No near-duplicates were found, and no teacher, SQL or generator item matched an evaluation item.

MBPP's train and test splits contain near-duplicate tasks (for example, "find the first repeated character in a given string" appears in both), which accounts for most MBPP drops.

## Statistics

- 6,570 examples, 1,671,657 tokens in total (training split: 1,587,255).
- 254 tokens per example on average, 904 maximum.

## Known limitations

- **Teacher answers are not human-verified.** They are filtered only by MCQ/TF key agreement and a manual spot check, and open-ended answers can still contain errors.
- **Possible Spider overlap:** `sql-create-context` may include Spider dev questions. Spider is not part of the current evaluation; if it is added, Spider-dev items should be removed from the training data first.
- **Generator coverage:** the generators cover a fixed set of problem types and phrasings. Gains on them measure procedural skill on those types, not general reasoning.
- **Licensing:** CS-Bench (CC BY-NC 4.0) is used for evaluation only and never for training.
