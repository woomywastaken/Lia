@echo off
rem Builds Lia.exe from lia_app.py, server.py, lia.html and the static folder.
rem
rem   build_exe.bat          ->  dist\Lia.exe  +  dist\models   (recommended: starts fast)
rem   build_exe.bat single   ->  dist\Lia.exe with the models packed inside (one big file, slow to start)
rem
rem Build it on Windows, with the same Python you run Lia with. LM Studio is not included: it stays its own program.
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat

set LIA_BUNDLE_MODELS=
if /i "%1"=="single" set LIA_BUNDLE_MODELS=1

python -m pip install pyinstaller
if errorlevel 1 goto failed
python -m PyInstaller Lia.spec --noconfirm --clean
if errorlevel 1 goto failed

if not defined LIA_BUNDLE_MODELS if exist models xcopy /E /I /Y /Q models dist\models >nul

echo.
echo Done. Your app is in the dist folder:
dir /b dist
echo.
echo Double-click dist\Lia.exe to start it (start LM Studio's server first).
if not defined LIA_BUNDLE_MODELS echo Keep the models folder next to Lia.exe. Hand in the whole dist folder.
pause
exit /b 0

:failed
echo.
echo The build failed. Scroll up to the first line that says ERROR and send it to Claude.
pause
exit /b 1
