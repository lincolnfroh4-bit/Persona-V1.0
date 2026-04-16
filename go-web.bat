@echo off
setlocal

if exist runtime\python.exe (
    set "PYTHON_CMD=runtime\python.exe"
) else (
    if exist .venv\Scripts\python.exe (
        set "PYTHON_CMD=.venv\Scripts\python.exe"
    ) else (
        set "PYTHON_CMD=python"
    )
)

"%PYTHON_CMD%" infer-web.py --pycmd "%PYTHON_CMD%" --port 3000
pause
