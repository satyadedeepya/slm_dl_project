@echo off
rem FT-E from the training step onward (the v2 mix is built by data/build_explain_stage.py expand + mix).
cd /d %~dp0..
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -u train\train_lora.py --run-name ft-e --init-adapter outputs\ft-d\adapter --lr 1e-4 --epochs 1 --batch-size 1 --grad-accum 32 --train-file data\external\sft\explain_stage_v2_train.jsonl --val-file data\external\sft\explain_stage_v2_val.jsonl --val-limit 60 --eval-steps 100 > outputs\ft-e.log 2>&1 || exit /b 1
set EVAL=.venv\Scripts\python -u eval\run_eval.py --run-name FT-E --adapter outputs\ft-e\adapter
%EVAL% --batch-size 32 --tasks gen_dl gen_dbms gen_cn > outputs\eval-ft-e.log 2>&1
%EVAL% --batch-size 16 --tasks spider >> outputs\eval-ft-e.log 2>&1
%EVAL% --batch-size 32 --tasks mbpp >> outputs\eval-ft-e.log 2>&1
.venv\Scripts\python eval\explain_eval.py answer --model ft-e > outputs\explain-ft-e.log 2>&1
.venv\Scripts\python eval\explain_eval.py judge --a base --b ft-e --workers 5 >> outputs\explain-ft-e.log 2>&1
.venv\Scripts\python eval\explain_eval.py report --a base --b ft-e >> outputs\explain-ft-e.log 2>&1
echo done > outputs\fte.done
