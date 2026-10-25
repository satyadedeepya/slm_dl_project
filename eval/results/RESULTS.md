# Evaluation results

Accuracy in %, with the 95% bootstrap confidence interval in brackets.

| Task | n | B0_zeroshot | B1_3shot | FT-A | FT-B | FT-C | FT-D | FT-E |
|---|---|---|---|---|---|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | 400 | 51.2 [46.8–56.0] | 58.0 [53.0–63.0] | 53.8 [48.8–58.8] | 55.2 [50.5–60.2] | – | – | – |
| Verified OS numericals (exact match) | 100 | 4.0 [1.0–8.0] | 4.0 [1.0–8.0] | 3.0 [0.0–7.0] | 5.0 [1.0–9.0] | – | – | – |
| Verified CN numericals (exact match) | 100 | 1.0 [0.0–3.0] | 2.0 [0.0–5.0] | 0.0 [0.0–0.0] | 8.0 [3.0–13.0] | 7.0 [2.0–12.0] | 12.0 [6.0–19.0] | 11.0 [5.0–17.0] |
| Verified DBMS FD problems (exact match) | 100 | 15.0 [8.0–23.0] | 12.0 [6.0–19.0] | 17.0 [11.0–25.0] | 30.0 [21.0–40.0] | 31.0 [22.0–41.0] | 24.0 [16.0–33.0] | 35.0 [26.0–45.0] |
| Verified DL numericals (exact match) | 100 | 2.0 [0.0–5.0] | – | – | 3.0 [0.0–7.0] | 5.0 [1.0–9.0] | 21.0 [13.0–29.0] | 13.0 [7.0–20.0] |
| MBPP (pass@1) | 200 | 41.5 [34.5–48.0] | 41.5 [35.0–48.0] | 41.0 [34.5–47.5] | 42.5 [36.0–49.0] | 40.0 [34.0–47.0] | 39.5 [33.0–46.5] | 40.0 [33.5–47.0] |
| Spider dev (execution acc.) | 200 | 50.5 [44.0–57.5] | – | 54.5 [48.0–61.0] | 60.0 [53.0–66.5] | 59.5 [52.5–66.0] | 58.5 [52.0–65.0] | 59.0 [52.5–66.0] |
| GATE MCQ | 56 | 33.9 [21.4–46.4] | – | 37.5 [25.0–50.0] | 33.9 [21.4–46.4] | – | – | – |

## Paired significance vs B0_zeroshot

Exact McNemar test on the same items: how many items only the reference got right, how many only the other run got right, and the two-sided p-value.

| Task | Run | Only ref. correct | Only run correct | p |
|---|---|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | B1_3shot | 42 | 69 | 0.0132 |
| CS-Bench (DSA/OS/CN, MCQ + T/F) | FT-A | 56 | 66 | 0.4153 |
| CS-Bench (DSA/OS/CN, MCQ + T/F) | FT-B | 55 | 71 | 0.1812 |
| Verified OS numericals (exact match) | B1_3shot | 3 | 3 | 1.0000 |
| Verified OS numericals (exact match) | FT-A | 4 | 3 | 1.0000 |
| Verified OS numericals (exact match) | FT-B | 2 | 3 | 1.0000 |
| Verified CN numericals (exact match) | B1_3shot | 1 | 2 | 1.0000 |
| Verified CN numericals (exact match) | FT-A | 1 | 0 | 1.0000 |
| Verified CN numericals (exact match) | FT-B | 1 | 8 | 0.0391 |
| Verified CN numericals (exact match) | FT-C | 1 | 7 | 0.0703 |
| Verified CN numericals (exact match) | FT-D | 1 | 12 | 0.0034 |
| Verified CN numericals (exact match) | FT-E | 1 | 11 | 0.0063 |
| Verified DBMS FD problems (exact match) | B1_3shot | 11 | 8 | 0.6476 |
| Verified DBMS FD problems (exact match) | FT-A | 6 | 8 | 0.7905 |
| Verified DBMS FD problems (exact match) | FT-B | 10 | 25 | 0.0167 |
| Verified DBMS FD problems (exact match) | FT-C | 5 | 21 | 0.0025 |
| Verified DBMS FD problems (exact match) | FT-D | 8 | 17 | 0.1078 |
| Verified DBMS FD problems (exact match) | FT-E | 7 | 27 | 0.0008 |
| Verified DL numericals (exact match) | FT-B | 1 | 2 | 1.0000 |
| Verified DL numericals (exact match) | FT-C | 2 | 5 | 0.4531 |
| Verified DL numericals (exact match) | FT-D | 1 | 20 | 0.0000 |
| Verified DL numericals (exact match) | FT-E | 2 | 13 | 0.0074 |
| MBPP (pass@1) | B1_3shot | 22 | 22 | 1.0000 |
| MBPP (pass@1) | FT-A | 26 | 25 | 1.0000 |
| MBPP (pass@1) | FT-B | 25 | 27 | 0.8899 |
| MBPP (pass@1) | FT-C | 31 | 28 | 0.7948 |
| MBPP (pass@1) | FT-D | 30 | 26 | 0.6889 |
| MBPP (pass@1) | FT-E | 30 | 27 | 0.7914 |
| Spider dev (execution acc.) | FT-A | 24 | 32 | 0.3497 |
| Spider dev (execution acc.) | FT-B | 16 | 35 | 0.0110 |
| Spider dev (execution acc.) | FT-C | 16 | 34 | 0.0153 |
| Spider dev (execution acc.) | FT-D | 16 | 32 | 0.0293 |
| Spider dev (execution acc.) | FT-E | 15 | 32 | 0.0186 |
| GATE MCQ | FT-A | 7 | 9 | 0.8036 |
| GATE MCQ | FT-B | 9 | 9 | 1.0000 |

## CS-Bench (DSA/OS/CN, MCQ + T/F): breakdown

| Group | n | B0_zeroshot | B1_3shot | FT-A | FT-B | FT-C | FT-D | FT-E |
|---|---|---|---|---|---|---|---|---|
| Assertion | 100 | 51.0 | 60.0 | 61.0 | 62.0 | – | – | – |
| CN | 138 | 55.1 | 61.6 | 56.5 | 57.2 | – | – | – |
| DSA | 144 | 56.9 | 57.6 | 55.6 | 57.6 | – | – | – |
| Knowledge | 258 | 54.7 | 63.6 | 62.4 | 64.3 | – | – | – |
| Multiple-choice | 300 | 51.3 | 57.3 | 51.3 | 53.0 | – | – | – |
| OS | 118 | 39.8 | 54.2 | 48.3 | 50.0 | – | – | – |
| Reasoning | 142 | 45.1 | 47.9 | 38.0 | 38.7 | – | – | – |

## Verified OS numericals (exact match): breakdown

| Group | n | B0_zeroshot | B1_3shot | FT-A | FT-B | FT-C | FT-D | FT-E |
|---|---|---|---|---|---|---|---|---|
| cpu_scheduling | 43 | 0.0 | 0.0 | 0.0 | 0.0 | – | – | – |
| deadlock | 12 | 25.0 | 33.3 | 0.0 | 25.0 | – | – | – |
| disk_scheduling | 18 | 0.0 | 0.0 | 0.0 | 0.0 | – | – | – |
| page_replacement | 27 | 3.7 | 0.0 | 11.1 | 7.4 | – | – | – |

## Verified CN numericals (exact match): breakdown

| Group | n | B0_zeroshot | B1_3shot | FT-A | FT-B | FT-C | FT-D | FT-E |
|---|---|---|---|---|---|---|---|---|
| checksum | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| crc | 26 | 0.0 | 0.0 | 0.0 | 7.7 | 7.7 | 3.8 | 0.0 |
| delay | 14 | 7.1 | 0.0 | 0.0 | 7.1 | 7.1 | 0.0 | 21.4 |
| flow_control | 17 | 0.0 | 5.9 | 0.0 | 11.8 | 5.9 | 17.6 | 17.6 |
| subnetting | 33 | 0.0 | 3.0 | 0.0 | 9.1 | 9.1 | 24.2 | 15.2 |

## Verified DBMS FD problems (exact match): breakdown

| Group | n | B0_zeroshot | B1_3shot | FT-A | FT-B | FT-C | FT-D | FT-E |
|---|---|---|---|---|---|---|---|---|
| decomposition | 16 | 50.0 | 31.2 | 75.0 | 43.8 | 68.8 | 37.5 | 37.5 |
| fd | 61 | 3.3 | 6.6 | 0.0 | 32.8 | 27.9 | 16.4 | 37.7 |
| normalization | 23 | 21.7 | 13.0 | 21.7 | 13.0 | 13.0 | 34.8 | 26.1 |

## Verified DL numericals (exact match): breakdown

| Group | n | B0_zeroshot | B1_3shot | FT-A | FT-B | FT-C | FT-D | FT-E |
|---|---|---|---|---|---|---|---|---|
| attention | 6 | 0.0 | – | – | 0.0 | 0.0 | 0.0 | 0.0 |
| backprop | 13 | 0.0 | – | – | 0.0 | 0.0 | 7.7 | 0.0 |
| cnn | 22 | 0.0 | – | – | 0.0 | 13.6 | 45.5 | 27.3 |
| forward | 14 | 0.0 | – | – | 0.0 | 0.0 | 7.1 | 0.0 |
| loss | 8 | 0.0 | – | – | 0.0 | 0.0 | 0.0 | 0.0 |
| metrics | 9 | 0.0 | – | – | 0.0 | 0.0 | 0.0 | 0.0 |
| mlp | 11 | 0.0 | – | – | 0.0 | 18.2 | 72.7 | 63.6 |
| optim | 17 | 11.8 | – | – | 17.6 | 0.0 | 5.9 | 0.0 |

## GATE MCQ: breakdown

| Group | n | B0_zeroshot | B1_3shot | FT-A | FT-B | FT-C | FT-D | FT-E |
|---|---|---|---|---|---|---|---|---|
| CN | 10 | 50.0 | – | 20.0 | 40.0 | – | – | – |
| DBMS | 9 | 33.3 | – | 33.3 | 55.6 | – | – | – |
| DSA | 32 | 31.2 | – | 46.9 | 28.1 | – | – | – |
| OS | 5 | 20.0 | – | 20.0 | 20.0 | – | – | – |

## Answer-format failures

Answers with no parseable choice, and accuracy on the parseable ones.

| Task | B0_zeroshot | B1_3shot | FT-A | FT-B | FT-C | FT-D | FT-E |
|---|---|---|---|---|---|---|---|
| CS-Bench (DSA/OS/CN, MCQ + T/F) | 21/400 unparsed; 54.1 on parsed | 4/400 unparsed; 58.6 on parsed | 4/400 unparsed; 54.3 on parsed | 7/400 unparsed; 56.2 on parsed | – | – | – |
| GATE MCQ | 7/56 unparsed; 38.8 on parsed | – | 5/56 unparsed; 41.2 on parsed | 6/56 unparsed; 38.0 on parsed | – | – | – |

## Run settings

| Run | Model | Adapter | Few-shot |
|---|---|---|---|
| B0_zeroshot | Qwen/Qwen2.5-1.5B-Instruct | – | 0 |
| B1_3shot | Qwen/Qwen2.5-1.5B-Instruct | – | 3 |
| FT-A | Qwen/Qwen2.5-1.5B-Instruct | outputs\ft-a\adapter | 0 |
| FT-B | Qwen/Qwen2.5-1.5B-Instruct | outputs\ft-b\adapter | 0 |
| FT-C | Qwen/Qwen2.5-1.5B-Instruct | outputs\ft-c\adapter | 0 |
| FT-D | Qwen/Qwen2.5-1.5B-Instruct | outputs\ft-d\adapter | 0 |
| FT-E | Qwen/Qwen2.5-1.5B-Instruct | outputs\ft-e\adapter | 0 |
