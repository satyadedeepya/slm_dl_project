# Evaluation results

Accuracy in %, with the 95% bootstrap confidence interval in brackets.

| Task | n | B0_zeroshot | B1_3shot |
|---|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | 400 | 51.2 [46.8–56.0] | 58.0 [53.0–63.0] |
| Verified OS numericals (exact match) | 100 | 4.0 [1.0–8.0] | 4.0 [1.0–8.0] |
| MBPP (pass@1) | 200 | 41.5 [34.5–48.0] | 41.5 [35.0–48.0] |

## Paired significance vs B0_zeroshot

Exact McNemar test on the same items: how many items only the reference got right, how many only the other run got right, and the two-sided p-value.

| Task | Run | Only ref. correct | Only run correct | p |
|---|---|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | B1_3shot | 42 | 69 | 0.0132 |
| Verified OS numericals (exact match) | B1_3shot | 3 | 3 | 1.0000 |
| MBPP (pass@1) | B1_3shot | 22 | 22 | 1.0000 |

## CS-Bench (DSA/OS/CN, MCQ + T/F): breakdown

| Group | n | B0_zeroshot | B1_3shot |
|---|---|---|---|
| Assertion | 100 | 51.0 | 60.0 |
| CN | 138 | 55.1 | 61.6 |
| DSA | 144 | 56.9 | 57.6 |
| Knowledge | 258 | 54.7 | 63.6 |
| Multiple-choice | 300 | 51.3 | 57.3 |
| OS | 118 | 39.8 | 54.2 |
| Reasoning | 142 | 45.1 | 47.9 |

## Verified OS numericals (exact match): breakdown

| Group | n | B0_zeroshot | B1_3shot |
|---|---|---|---|
| cpu_scheduling | 43 | 0.0 | 0.0 |
| deadlock | 12 | 25.0 | 33.3 |
| disk_scheduling | 18 | 0.0 | 0.0 |
| page_replacement | 27 | 3.7 | 0.0 |

## Run settings

| Run | Model | Adapter | Few-shot |
|---|---|---|---|
| B0_zeroshot | Qwen/Qwen2.5-1.5B-Instruct | – | 0 |
| B1_3shot | Qwen/Qwen2.5-1.5B-Instruct | – | 3 |
