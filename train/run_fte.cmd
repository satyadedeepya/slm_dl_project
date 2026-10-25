@echo off
rem FT-E: FT-C + explanation stage with fuller (250-350 word) teacher answers, then evaluation.
rem   1. finish the FT-D vs base explanation judge (network only, runs alongside)
rem   2. expand short teacher answers (data/build_explain_stage.py expand) and build the v2 mix
rem   3. train FT-E from FT-C's adapter
rem   4. accuracy on the FT-C task set, then blind explanation comparison against the base model
cd /d %~dp0..
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
start "judge-ftd" /b cmd /c "eval\run_explain_judge_ftd.cmd > outputs\explain-ft-d-judge.log 2>&1"
.venv\Scripts\python data\build_explain_stage.py expand --workers 6 > outputs\explain_expand.log 2>&1 || exit /b 1
.venv\Scripts\python data\build_explain_stage.py mix --tag explain_stage_v2 >> outputs\explain_expand.log 2>&1 || exit /b 1
.venv\Scripts\python -u train\train_lora.py --run-name ft-e --init-adapter outputs\ft-c\adapter --lr 1e-4 --epochs 1 --batch-size 1 --grad-accum 32 --train-file data\external\sft\explain_stage_v2_train.jsonl --val-file data\external\sft\explain_stage_v2_val.jsonl --val-limit 100 --eval-steps 27 > outputs\ft-e.log 2>&1 || exit /b 1
set EVAL=.venv\Scripts\python -u eval\run_eval.py --run-name FT-E --adapter outputs\ft-e\adapter
%EVAL% --batch-size 32 --tasks gen_dl gen_dbms gen_cn > outputs\eval-ft-e.log 2>&1
%EVAL% --batch-size 16 --tasks spider >> outputs\eval-ft-e.log 2>&1
%EVAL% --batch-size 32 --tasks mbpp >> outputs\eval-ft-e.log 2>&1
.venv\Scripts\python eval\explain_eval.py answer --model ft-e > outputs\explain-ft-e.log 2>&1
.venv\Scripts\python eval\explain_eval.py judge --a base --b ft-e >> outputs\explain-ft-e.log 2>&1
.venv\Scripts\python eval\explain_eval.py report --a base --b ft-e >> outputs\explain-ft-e.log 2>&1
echo done > outputs\fte.done
