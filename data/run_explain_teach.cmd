@echo off
rem Teacher answers for the FT-D explanation stage (Gemini; cached and resumable), then the training mix.
cd /d %~dp0..
.venv\Scripts\python data\build_explain_stage.py teach || exit /b 1
.venv\Scripts\python data\build_explain_stage.py mix
echo done > outputs\explain_teach.done
