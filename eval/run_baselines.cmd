@echo off
rem Runs the Week-1 baselines back to back. Logs go to eval\results\<run>.log
cd /d %~dp0..
set HF_HUB_DISABLE_XET=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -u eval\run_eval.py --run-name B0_zeroshot --batch-size 32 --tasks csbench gen_os mbpp > eval\results\B0_zeroshot.log 2>&1
.venv\Scripts\python -u eval\run_eval.py --run-name B1_3shot --fewshot 3 --batch-size 16 --tasks csbench gen_os mbpp > eval\results\B1_3shot.log 2>&1
echo done > eval\results\baselines.done
