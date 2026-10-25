@echo off
rem Finish the blind explanation judge for FT-D vs the base model (answers already generated).
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
.venv\Scripts\python eval\explain_eval.py judge --a base --b ft-d --workers 2 || exit /b 1
.venv\Scripts\python eval\explain_eval.py report --a base --b ft-d
echo done > outputs\explain_ftd.done
