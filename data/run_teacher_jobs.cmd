@echo off
rem Runs all Gemini (teacher) data jobs in priority order. Each job caches its progress and stops
rem cleanly when the free-tier daily quota is used up, so run this once a day until everything is done.
rem   1. GATE explanations (verified against the official key)
rem   2. textbook-grounded Q&A
rem   3. remaining concept Q&A topics
rem Then rebuild the SFT dataset.
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
.venv\Scripts\python -u data\build_gate.py --explain
.venv\Scripts\python -u data\build_textbook_qa.py
.venv\Scripts\python -u data\build_concepts.py
.venv\Scripts\python -u data\build_dataset.py
