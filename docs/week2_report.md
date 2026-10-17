# Week 2 Report: Verified Data and the SFT Dataset

**Week 2 goal (from the plan):** CN and DBMS generators, concept Q&A, merge public datasets, dedupe, decontaminate, and produce the training split.

## 1. Summary

| Deliverable | Status |
|---|---|
| CN generator: subnetting, CRC, checksum, delays, sliding-window utilisation | ✅ 900 train / 100 held-out |
| DBMS generator: closure, candidate keys, highest normal form, lossless join | ✅ 600 train / 100 held-out |
| Grader extended: tokens (IPs, bit strings, hex), relative tolerance, attribute sets, key sets, choices | ✅ |
| Tests: textbook examples + grader edge cases + answer-balance checks | ✅ 36 passing |
| Concept Q&A pipeline (Gemini teacher, 133 topics) | ✅ 1,183 items from 108 topics; 25 topics pending quota |
| Public data: MBPP, sql-create-context, CodeAlpaca, alpaca-cleaned | ✅ |
| Dataset build: dedupe, 13-gram decontamination, length filter, split | ✅ 6,241 train / 329 val |
| Baselines (B0/B1) on the new CN and DBMS test sets | ✅ |

## 2. New verified generators

| Subject | Family | How the answer is computed | Test split differs by |
|---|---|---|---|
| CN | Subnetting (analyse / divide) | Python `ipaddress` | wider prefix and subnet-count ranges |
| CN | CRC sender / receiver | mod-2 long division, every XOR step shown | longer data words |
| CN | Internet checksum | 16-bit one's complement sum with end-around carry | more words |
| CN | Delays (single link, store-and-forward) | closed form, ms | disjoint parameter values |
| CN | Utilisation (stop-and-wait, GBN, SR, minimum window + sequence bits) | 1/(1+2a), min(1, W/(1+2a)) | disjoint parameter values |
| DBMS | Attribute closure | fixed-point closure with the FD firing order | larger schemas |
| DBMS | Candidate keys | minimal superkeys by search, with the "never on a right-hand side" shortcut | larger schemas |
| DBMS | Highest normal form | exact 2NF (closure of key subsets), 3NF and BCNF tests; answers balanced across 1NF–BCNF | larger schemas |
| DBMS | Lossless join (binary) | (R1∩R2)+ ⊇ R1 or R2; balanced lossless/lossy | larger schemas |

**Validation:**
- **Textbook cases:** Forouzan CRC (1001 / 1011 → 110), Tanenbaum CRC (1101011111 / 10011 → 11010111110010), the IPv4 header checksum example (→ B861), /26 and /21 subnet cases, and classic FD examples (AB→C, C→D, D→A has keys AB, BC, BD and is 3NF).
- **Self-consistency:** 6,000 reference solutions pass the grader.
- **Grader edge cases:** a CRC remainder only counts as a whole token, so "110" inside "1001110" is rejected; "it is not in BCNF, so the highest is 3NF" is graded as 3NF.

**Train/test overlap:** in the first version, 2 CN test questions also appeared in training, because the delay and window problems drew from small value lists. The test split now uses disjoint parameter values, and the overlap is 0.

## 3. Concept Q&A (teacher model)

- **Topics:** 133 syllabus topics (`data/concepts/topics.json`): DSA 30, Programming 25, OS 26, CN 26, DBMS 26. Each topic gets 6 open-ended, 3 MCQ and 2 true/false items.
- **Generation:** questions and answers are generated in **separate requests**, as GLAN recommends, with 3 topics per request.
- **Filter:** MCQ/TF items are kept only if the answer's final line matches the key written with the question. 5 items were rejected this way.
- **Free-tier limits:** the Gemini key allows **20 requests per day per model** (quota `GenerateRequestsPerDayPerProjectPerModel-FreeTier`). The pipeline therefore:
  - batches topics,
  - moves to the next Flash model when a model's daily quota is used up or it fails 3 times in a row (503 overloads, truncated JSON),
  - caches every finished topic, so a run resumes where it stopped.
- **Teacher mix:** gemini-3.8-flash 429 items, gemini-3.5-flash-lite 393, gemini-3-flash-preview 263, gemini-3.6-flash 98. The model is recorded per item, so its effect can be analysed later.
- **Pending:** 25 topics (22 DBMS, 3 CN TCP topics). One batch's answers were lost to an id mismatch, so those topics were re-queued. They need one more day of quota; then `python data/build_concepts.py` and `python data/build_dataset.py` add them.
- **Spot check:** sampled answers from both the Flash and Flash-Lite models were correct (for example, the Θ-notation proof constants and the Pure ALOHA vulnerable period of 2T). Flash-Lite answers are noticeably shorter.

## 4. The SFT dataset

| Source | Subject(s) | Items | Tokens | Quality gate |
|---|---|---|---|---|
| generator | OS 900, CN 900, DBMS 600 | 2,400 | 884k | computed answers |
| teacher | DSA 330, OS 284, Prog 274, CN 252, DBMS 43 | 1,183 | 251k | key agreement (MCQ/TF) |
| sql | DBMS | 1,000 | 161k | SQLite accepts the query |
| codealpaca | Programming | 1,176 | 190k | `ast.parse` |
| mbpp | Programming | 311 | 79k | reference code passes its tests |
| replay | General | 500 | 107k | length 80–1500 chars |
| **total** | | **6,570** | **1.67M** | |

- **Split:** 6,241 train / 329 validation, 5% stratified by source.
- **Length:** mean 254 tokens, maximum 904 (limit 1,024).
- **Decontamination:**
  - 62 MBPP and 24 CodeAlpaca items were dropped for sharing a 13-gram with an evaluation item. MBPP's own train and test splits contain near-duplicate tasks.
  - **No teacher item overlapped CS-Bench**, which matters because the concept questions are closest to CS-Bench's style.

**Imbalance to watch:** generator problems are 37% of items but **53% of tokens**, because their solutions are long tables and traces. If the fine-tuned model overfits to that format, Week 3 can cap the generator slice.

## 5. Baselines on the new held-out tests

| Task | B0 zero-shot | B1 3-shot | McNemar p |
|---|---|---|---|
| CN numericals | 1.0 [0–3] | 2.0 [0–5] | 1.00 |
| DBMS FD problems | 15.0 [8–23] | 12.0 [6–19] | 0.65 |

**Breakdown (B0):**
- CN: subnetting 0/33, CRC 0/26, checksum 0/10, flow control 0/17, delay 1/14.
- DBMS: closure and keys 2/61 (3%), normal form 5/23 (22%), lossless join 8/16 (50%).

**Interpretation:** the DBMS "successes" are at chance level. Lossless join is a two-way choice (50% expected by guessing) and normal form a four-way choice (25% expected). On the items that require an actual computation (closures and candidate keys), the model scores 3%. As with OS, 3-shot examples do not help. Together with Week 1, the base model is at or near **zero on every verified procedural task in OS, CN and DBMS**.

## 6. Additional data sources (added after the first build)

To use as many accurate sources as possible, three more were added.

**Spider (text-to-SQL):** replaces `sql-create-context`.
- 1,335 Spider train questions, each checked by running the gold query on its real SQLite database.
- Spider dev becomes an evaluation set. sql-create-context may contain dev questions, which is why it was replaced.

**GATE CSE previous-year questions:** 1,210 single-answer MCQs in our five subjects, from the GATE Overflow archive (pinned commit). Answers are official keys or GATE Overflow-verified.
- **Split:** all papers ≤ 2020 → training; 2021–2026 split 50/50 by year and subject. That gives 1,154 for training and 56 for a held-out test.
- **Explanations:** Gemini solves each training question without seeing the key, and only explanations that reach the key are kept.
- **Copyright:** GATE text is copyrighted, so it is downloaded at build time into `data/external/` (git-ignored). Only question IDs are committed.

**Open textbooks:** 240 passages (60 per book). Gemini writes Q&A answerable from each passage alone, and every item records its book and license.
- Peterson & Davie, *Computer Networks* (CC BY 4.0)
- Hailperin, *Operating Systems and Middleware* (CC BY-SA 3.0; its transactions chapter counts as DBMS)
- Erickson, *Algorithms* (CC BY 4.0)
- Morin, *Open Data Structures* (CC BY 2.5)

Text extraction repairs PDF glyph names and ligatures, and skips exercise-heavy passages. Copyrighted standard texts are not used. Watt's *Database Design* was left out because its hosts block automated download.

**New zero-shot baselines:**

| Task | n | B0 |
|---|---|---|
| Spider dev (execution accuracy) | 200 | 50.5 [44.0–57.5] |
| GATE test (MCQ) | 56 | 33.9 [21.4–46.4] |

GATE by subject: CN 5/10, DBMS 3/9, DSA 10/32, OS 1/5. Chance on four options is 25%. These two tasks have no few-shot (B1) variant yet.

**Pending (Gemini free quota):**
- GATE explanations: about 96 requests
- Textbook Q&A: about 60 requests
- The last 25 concept topics: about 9 requests

`data/run_teacher_jobs.cmd` runs all three in priority order and rebuilds the dataset. Each job resumes where it stopped.

## 7. Plan for Week 3

- **FT-B (full mix)** vs **FT-A (no generator slice)**, both evaluated with the same harness against B1, the stronger baseline. Reporting covers McNemar p-values, format-failure rates and per-family accuracy.
- **Training budget:** about 1.59M training tokens per epoch (FT-A: about 0.70M).
- **Compute:**
  - On a T4 with Unsloth, this is roughly 1 hour per run (estimate).
  - On the local Arc iGPU, 4-bit QLoRA (bitsandbytes) is not dependable, so it would be plain LoRA in fp16 at several hours per epoch (unmeasured).
  - A short timing run will decide where training happens.
