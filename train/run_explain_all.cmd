@echo off
rem Whole FT-D pipeline, unattended:
rem   teacher answers (Gemini, network)  in parallel with  base/FT-C explanation baseline (GPU, then Gemini judge)
rem   then FT-D training and evaluation (train\run_ftd.cmd).
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
start "explain-teach" /b cmd /c "data\run_explain_teach.cmd > outputs\explain_teach.log 2>&1"
un_explain_teach.cmd > outputsexplain_teach.log 2>&1"
call eval\run_explain_baseline.cmd > outputs\explain_baseline.log 2>&1
:wait
if exist outputs\explain_teach.done goto train
timeout /t 30 /nobreak > nul
goto wait
:train
call train\run_ftd.cmd
