@echo off
rem Remaining Week 3 steps: finish FT-A (resumes from its latest checkpoint), then evaluate it.
cd /d %~dp0..
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set TRAIN=--epochs 1 --batch-size 1 --grad-accum 32 --val-limit 150 --eval-steps 35
.venv\Scripts\python -u train\train_lora.py --run-name ft-a --exclude-source generator %TRAIN% >> outputs\ft-a.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name FT-A --adapter outputs\ft-a\adapter --batch-size 32 --tasks csbench gen_os gen_cn gen_dbms mbpp gate > outputs\eval-ft-a.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name FT-A --adapter outputs\ft-a\adapter --batch-size 16 --tasks spider >> outputs\eval-ft-a.log 2>&1
echo done > outputs\week3.done
