@echo off
rem Teacher answers for the FT-D explanation stage (Gemini; cached and resumable), then the training mix.
cd /d %~dp0..
.venv\Scripts\python data\build_explain_stage.py teach --models gemini-3.5-flash-lite gemini-3.1-flash-lite gemini-flash-lite-latest gemini-2.5-flash-lite gemini-3.1-flash-lite-preview gemma-4-31b-it gemma-4-26b-a4b-it || exit /b 1
.venv\Scripts\python data\build_explain_stage.py mix
echo done > outputs\explain_teach.done
