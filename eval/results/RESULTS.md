# Evaluation results

Accuracy in %, with the 95% bootstrap confidence interval in brackets.

| Task | n | B0_zeroshot | B1_3shot |
|---|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | 400 | 51.2 [46.8–56.0] | 58.0 [53.0–63.0] |
| Verified OS numericals (exact match) | 100 | 4.0 [1.0–8.0] | 4.0 [1.0–8.0] |
| Verified CN numericals (exact match) | 100 | 1.0 [0.0–3.0] | 2.0 [0.0–5.0] |
| Verified DBMS FD problems (exact match) | 100 | 15.0 [8.0–23.0] | 12.0 [6.0–19.0] |
| MBPP (pass@1) | 200 | 41.5 [34.5–48.0] | 41.5 [35.0–48.0] |
| Spider dev (execution acc.) | 200 | 50.5 [44.0–57.5] | – |
| GATE MCQ | 56 | 33.9 [21.4–46.4] | – |

## Paired significance vs B0_zeroshot

Exact McNemar test on the same items: how many items only the reference got right, how many only the other run got right, and the two-sided p-value.

| Task | Run | Only ref. correct | Only run correct | p |
|---|---|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | B1_3shot | 42 | 69 | 0.0132 |
| Verified OS numericals (exact match) | B1_3shot | 3 | 3 | 1.0000 |
| Verified CN numericals (exact match) | B1_3shot | 1 | 2 | 1.0000 |
| Verified DBMS FD problems (exact match) | B1_3shot | 11 | 8 | 0.6476 |
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

## Verified CN numericals (exact match): breakdown

| Group | n | B0_zeroshot | B1_3shot |
|---|---|---|---|
| checksum | 10 | 0.0 | 0.0 |
| crc | 26 | 0.0 | 0.0 |
| delay | 14 | 7.1 | 0.0 |
| flow_control | 17 | 0.0 | 5.9 |
| subnetting | 33 | 0.0 | 3.0 |

## Verified DBMS FD problems (exact match): breakdown

| Group | n | B0_zeroshot | B1_3shot |
|---|---|---|---|
| decomposition | 16 | 50.0 | 31.2 |
| fd | 61 | 3.3 | 6.6 |
| normalization | 23 | 21.7 | 13.0 |

## GATE MCQ: breakdown

| Group | n | B0_zeroshot | B1_3shot |
|---|---|---|---|
| CN | 10 | 50.0 | – |
| DBMS | 9 | 33.3 | – |
| DSA | 32 | 31.2 | – |
| OS | 5 | 20.0 | – |

## Answer-format failures

Answers with no parseable choice, and accuracy on the parseable ones.

| Task | B0_zeroshot | B1_3shot |
|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | 21/400 unparsed; 54.1 on parsed | 4/400 unparsed; 58.6 on parsed |
| GATE MCQ | 7/56 unparsed; 38.8 on parsed | – |

## Run settings

| Run | Model | Adapter | Few-shot |
|---|---|---|---|
| B0_zeroshot | Qwen/Qwen2.5-1.5B-Instruct | – | 0 |
| B1_3shot | Qwen/Qwen2.5-1.5B-Instruct | – | 3 |
