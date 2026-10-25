@echo off
rem Explanation-quality baseline: answers from the base model and FT-C (batched locally), then the Gemini judge.
rem Answers are generated locally in batches (stop the chat server first: it holds the GPU).
cd /d %~dp0..
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python eval\explain_eval.py answer --model base || exit /b 1
.venv\Scripts\python eval\explain_eval.py answer --model ft-c || exit /b 1
.venv\Scripts\python eval\explain_eval.py judge --a base --b ft-c || exit /b 1
.venv\Scripts\python eval\explain_eval.py report --a base --b ft-c
echo done > outputs\explain_baseline.done
