@echo off
rem Runs the baselines (B0 zero-shot, B1 3-shot) on every task. Logs go to eval\results\<run>*.log
rem Batch sizes fit a 16 GB shared-memory Intel Arc iGPU: 3-shot generator prompts (three worked
rem solutions each) run out of memory at batch 16, so they use batch 8.
cd /d %~dp0..
set HF_HUB_DISABLE_XET=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -u eval\run_eval.py --run-name B0_zeroshot --batch-size 32 --tasks csbench gen_os gen_cn gen_dbms mbpp > eval\results\B0_zeroshot.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name B1_3shot --fewshot 3 --batch-size 16 --tasks csbench mbpp > eval\results\B1_3shot.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name B1_3shot --fewshot 3 --batch-size 8 --tasks gen_os gen_cn gen_dbms > eval\results\B1_3shot_gen.log 2>&1
echo done > eval\results\baselines.done
