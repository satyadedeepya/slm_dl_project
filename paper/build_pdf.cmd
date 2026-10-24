@echo off
rem Render paper\paper.html to paper\CSE-Tutor_paper.pdf with headless Microsoft Edge (no LaTeX needed).
cd /d %~dp0
set EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
"%EDGE%" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="%~dp0CSE-Tutor_paper.pdf" "file:///%~dp0paper.html"
echo wrote %~dp0CSE-Tutor_paper.pdf
