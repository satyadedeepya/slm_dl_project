@echo off
rem FT-C: continue FT-B's adapter on the Deep Learning stage (data/build_dl_stage.py), then evaluate it.
rem Evaluation covers the new DL task and the tasks that show gains or forgetting most clearly
rem (csbench, gen_os and gate were skipped to save time).
cd /d %~dp0..
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -u train\train_lora.py --run-name ft-c --init-adapter outputs\ft-b\adapter --lr 1e-4 --epochs 1 --batch-size 1 --grad-accum 32 --train-file data\external\sft\dl_stage_train.jsonl --val-file data\external\sft\dl_stage_val.jsonl --val-limit 100 --eval-steps 22 > outputs\ft-c.log 2>&1
set EVAL=.venv\Scripts\python -u eval\run_eval.py --run-name FT-C --adapter outputs\ft-c\adapter
%EVAL% --batch-size 32 --tasks gen_dl gen_dbms gen_cn > outputs\eval-ft-c.log 2>&1
%EVAL% --batch-size 16 --tasks spider >> outputs\eval-ft-c.log 2>&1
%EVAL% --batch-size 32 --tasks mbpp >> outputs\eval-ft-c.log 2>&1
echo done > outputs\ftc.done
