@echo off
rem Starts Lia from the Python files (no .exe needed). Double-click it.
rem It connects to LM Studio by itself. For canned answers with no model:  start_lia.bat --demo
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python lia_app.py %*
if errorlevel 1 pause
