@echo off
rem Gemini/Gemma judge + report for base vs FT-C (answers already generated).
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
.venv\Scripts\python eval\explain_eval.py judge --a base --b ft-c || exit /b 1
.venv\Scripts\python eval\explain_eval.py report --a base --b ft-c
echo done > outputs\explain_baseline.done
