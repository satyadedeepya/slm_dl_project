@echo off
rem FT-D: continue FT-C's adapter on the explanation stage (data/build_explain_stage.py), then
rem (1) check that accuracy is kept on the tasks FT-C was evaluated on, and
rem (2) compare explanations blind against the base model and FT-C (eval/explain_eval.py).
cd /d %~dp0..
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
rem The chat server holds the GPU; stop it before training.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000 " ^| findstr LISTENING') do taskkill /PID %%p /F
.venv\Scripts\python -u train\train_lora.py --run-name ft-d --init-adapter outputs\ft-c\adapter --lr 1e-4 --epochs 1 --batch-size 1 --grad-accum 32 --train-file data\external\sft\explain_stage_train.jsonl --val-file data\external\sft\explain_stage_val.jsonl --val-limit 100 --eval-steps 27 > outputs\ft-d.log 2>&1 || exit /b 1
set EVAL=.venv\Scripts\python -u eval\run_eval.py --run-name FT-D --adapter outputs\ft-d\adapter
%EVAL% --batch-size 32 --tasks gen_dl gen_dbms gen_cn > outputs\eval-ft-d.log 2>&1
%EVAL% --batch-size 16 --tasks spider >> outputs\eval-ft-d.log 2>&1
%EVAL% --batch-size 32 --tasks mbpp >> outputs\eval-ft-d.log 2>&1
.venv\Scripts\python eval\explain_eval.py answer --model ft-d > outputs\explain-ft-d.log 2>&1
.venv\Scripts\python eval\explain_eval.py judge --a base --b ft-d >> outputs\explain-ft-d.log 2>&1
.venv\Scripts\python eval\explain_eval.py judge --a ft-c --b ft-d >> outputs\explain-ft-d.log 2>&1
.venv\Scripts\python eval\explain_eval.py report --a base --b ft-d >> outputs\explain-ft-d.log 2>&1
.venv\Scripts\python eval\explain_eval.py report --a ft-c --b ft-d >> outputs\explain-ft-d.log 2>&1
echo done > outputs\ftd.done
