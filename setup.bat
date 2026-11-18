@echo off
setlocal
REM ============================================================
REM  START ONLY (no install, no training).
REM  Run setup_all.bat once first.
REM ============================================================
set "ROOT=%~dp0"

if not exist "%ROOT%backend\venv\Scripts\activate.bat" (
  echo [ERROR] Backend is not set up yet. Run setup_all.bat first.
  pause
  exit /b 1
)
if not exist "%ROOT%frontend\node_modules" (
  echo [ERROR] Frontend is not set up yet. Run setup_all.bat first.
  pause
  exit /b 1
)
dir /b "%ROOT%backend\artifacts\model_v*.joblib" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] No trained model found in backend\artifacts. Run setup_all.bat first.
  pause
  exit /b 1
)

start "Fraud Backend" /d "%ROOT%backend" cmd /k "venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"
start "Fraud Frontend" /d "%ROOT%frontend" cmd /k "npm run dev"
timeout /t 6 >nul
start "" http://localhost:5173
exit /b 0
