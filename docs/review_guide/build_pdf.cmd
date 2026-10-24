@echo off
rem Render review_guide.html to docs\CSE-Tutor_Review_Guide.pdf with headless Microsoft Edge.
cd /d %~dp0
set EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
"%EDGE%" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=5000 --print-to-pdf="%~dp0..\CSE-Tutor_Review_Guide.pdf" "file:///%~dp0review_guide.html"
