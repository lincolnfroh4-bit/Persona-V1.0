@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "realtime_mic_voice_gui.py"
) else (
  python "realtime_mic_voice_gui.py"
)

if errorlevel 1 (
  echo.
  echo Realtime mic GUI exited with an error.
  pause
)
