# Explanation Quality Report: FT-D and FT-E

**Goal:** the tutor must give accurate steps *and* better explanations than the base model.

> Status: FT-D complete; FT-E results are in section 4.

## 1. The problem

FT-C was more accurate than the base model on numericals, but in the chat app its concept answers were shorter and less clear. A user comparison of "What is CNN?" showed this clearly.

| What we checked | Finding |
|---|---|
| Training target length | Teacher concept answers had a median of 57–91 words, and code/SQL targets 15–25 words. The base model writes about 290 words. |
| LaTeX in targets | 291 training answers contained `$$…$$`, which the chat app showed as raw symbols. |
| Blind judge, FT-C vs base | FT-C wins **numericals 14–4** (p = 0.03) but loses **concepts 6–13**: steps 2.7 vs 4.2, clarity 3.4 vs 4.3 (out of 5). |

Fine-tuning on short answers taught the model to answer briefly. It overwrote the base model's explanation style while adding accuracy.

## 2. The fixes

1. **Chat app:** math is shown as plain readable text, e.g. `z = Σ wᵢxᵢ + b`, `1.45 ≥ 0`, `O(n²)`. This is done by `demo/static/plainmath.js` and its Python twin `data/plain_math.py`, with tests. No `$$` or LaTeX ever reaches the student, and "Copy answer" copies the plain text.
2. **Explanation-style training data** (`data/build_explain_stage.py`):
   - About 1,340 questions: every open concept question, plus "What is X?" and "Explain X with an example." for all 169 syllabus topics.
   - Questions resembling the evaluation questions are removed (word-overlap ≥ 0.4).
   - A teacher model writes each answer in a fixed teaching style: a direct answer, numbered key points with bold labels, one worked example, and an "In short:" line.
   - All math is written as plain text, and 0 training targets contain LaTeX.
3. **FT-D:** continues FT-C on the explanation answers (1,192 answers, median 164 words) plus replay of verified problems (1,050) and code/SQL (250). 78 steps.
4. **FT-E:** continues FT-D on *fuller* answers. Teacher answers under 220 words were rewritten to 250–350 words, adding a "why/how" for each point, a worked numeric example and a common-mistake line.

## 3. Evaluation method (`eval/explain_eval.py`)

- **Questions:** 39 held-out questions.
  - 19 verified numericals (one per problem type across OS, CN, DBMS and DL), each with its reference solution.
  - 20 concept questions of the kind students type, e.g. "What is CNN?" and "Explain the TCP three-way handshake".
- **Answers:** generated greedily with the training system prompt and shown to the judge as the app shows them (plain-text math).
- **Judge:** one fixed judge model (Gemma 4 31B) compares two answers **blind**. The model names are hidden, and every pair is judged **twice with the order swapped**. A model wins a question only if it wins in both orders; otherwise the question counts as a tie.
- **Scores:** correctness, steps and clarity, each 1–5. Win/loss significance uses an exact sign test.

## 4. Results

### Explanation quality, blind pairwise against the base model

| Model vs base | Numericals: win–loss–tie (p) | Concepts: win–loss–tie (p) | Concept steps (model / base) | Concept clarity (model / base) | Median words |
|---|---|---|---|---|---|
| FT-C | 14–4–1 (**0.031**) | 6–13–1 (0.17) | 2.65 / 4.15 | 3.40 / 4.33 | 79 |
| FT-D | 13–4–2 (**0.049**) | 8–12–0 (0.50) | 3.17 / 3.95 | 3.85 / 4.28 | 102 |
| FT-E | **13–5–1** (p = 0.096) | 6–11–3 (p = 0.332) | 3.55 / 4.00 | 3.77 / 4.17 | 194 |

### Accuracy (exact match, %) on held-out tests

| Task | B0 base | FT-C | FT-D | FT-E |
|---|---|---|---|---|
| DL numericals (100) | 2 | 5 | **21** | 13 |
| DBMS FD problems (100) | 15 | 31 | 24 | 35 |
| CN numericals (100) | 1 | 7 | 12 | 11 |
| Spider SQL (200) | 50.5 | 59.5 | 58.5 | 59.0 |
| MBPP pass@1 (200) | 41.5 | 40.0 | 39.5 | 40.0 |

**FT-D vs FT-C (exact McNemar):**
- DL: 5 → 21, p < 0.001.
- DBMS: 31 → 24, p = 0.31.
- CN: 7 → 12, p = 0.23.
- Spider and MBPP: unchanged (p ≥ 0.75).

**FT-D vs base:**
- Significantly better on DL (p < 0.001), CN (p = 0.003) and Spider (p = 0.029).
- No significant difference on DBMS (p = 0.11) or MBPP (p = 0.69).

## 5. Analysis

1. **Accuracy:** every fine-tuned stage keeps a clear lead over the base model on numerical problems. FT-E wins the numericals 13–5 in the blind comparison, and its exact-match accuracy is significantly higher than the base model's on DL, DBMS, CN and Spider.
2. **Explanations:** answer length was the main cause of the explanation gap. Training targets grew from about 70 to 286 words, and the model's answers grew from 79 (FT-C) to 102 (FT-D) to 194 (FT-E) words. Concept results against the base model went from 6–13 (FT-C) to 8–12 (FT-D) to 6–11–3 (FT-E): concept correctness 3.1 vs 3.3, steps 3.5 vs 4.0, clarity 3.8 vs 4.2 (out of 5).
3. **Verdict:** FT-E narrows the explanation gap on concept questions, but the base model is still slightly ahead (6–11–3, p = 0.332, not a significant difference). Beating the base model clearly on open-ended explanations would need more or better teaching data, a second epoch, or preference tuning (DPO) with base-vs-tutor pairs.
4. **No LaTeX:** the chat app shows all math as plain text, whatever the model writes.
