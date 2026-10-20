@echo off
rem Week 3 protocol on the local Intel Arc iGPU: one epoch per run, evaluated on every task.
rem Training resumes from the latest checkpoint if interrupted; evaluation results are saved per task.
rem   1. FT-B: full mix          2. evaluate FT-B
rem   3. FT-A: no generator slice (ablation)   4. evaluate FT-A
cd /d %~dp0..
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set TRAIN=--epochs 1 --batch-size 1 --grad-accum 32 --val-limit 150 --eval-steps 35
.venv\Scripts\python -u train\train_lora.py --run-name ft-b %TRAIN% > outputs\ft-b.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name FT-B --adapter outputs\ft-b\adapter --batch-size 32 --tasks csbench gen_os gen_cn gen_dbms mbpp gate > outputs\eval-ft-b.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name FT-B --adapter outputs\ft-b\adapter --batch-size 16 --tasks spider >> outputs\eval-ft-b.log 2>&1
.venv\Scripts\python -u train\train_lora.py --run-name ft-a --exclude-source generator %TRAIN% > outputs\ft-a.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name FT-A --adapter outputs\ft-a\adapter --batch-size 32 --tasks csbench gen_os gen_cn gen_dbms mbpp gate > outputs\eval-ft-a.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name FT-A --adapter outputs\ft-a\adapter --batch-size 16 --tasks spider >> outputs\eval-ft-a.log 2>&1
echo done > outputs\week3.done
