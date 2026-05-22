@echo off
:: ════════════════════════════════════════════════════════════════
::  start.bat — Launch both the FastAPI backend and Streamlit UI
::  Run from the project root directory.
:: ════════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║   TrafficIQ — Starting up                   ║
echo  ║   FastAPI  : http://localhost:8000           ║
echo  ║   Streamlit: http://localhost:8501           ║
echo  ║   API Docs : http://localhost:8000/docs      ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: Use the project virtual environment so Streamlit and FastAPI share dependencies.
set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Project virtual environment not found at "%PYTHON_EXE%".
    echo Create it first, then run: venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

:: Start FastAPI in a new window
start "TrafficIQ API" cmd /k ""%PYTHON_EXE%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

:: Give the API a moment to load the model
timeout /t 5 /nobreak > nul

:: Start Streamlit in the current window
echo  Starting Streamlit dashboard...
"%PYTHON_EXE%" -m streamlit run app.py --server.port 8501
