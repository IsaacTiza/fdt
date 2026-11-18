@echo off
setlocal
REM ============================================================
REM  FIRST-TIME SETUP + TRAIN + START
REM  Put setup_all.bat and setup.bat in the project root (the
REM  folder containing "backend" and "frontend"), then double-click.
REM  Needs Python 3.10+ and Node.js 18+ installed.
REM  Dataset must be at: backend\data\PS_20174392719_1491204439457_log.csv
REM ============================================================
set "ROOT=%~dp0"
set "DATA=%ROOT%backend\data\PS_20174392719_1491204439457_log.csv"

echo === Checking prerequisites ===
where python >nul 2>nul
if errorlevel 1 ( echo [ERROR] Python not found on PATH. Install Python 3.10+ first. & goto :fail )
where node >nul 2>nul
if errorlevel 1 ( echo [ERROR] Node.js not found on PATH. Install Node 18+ first. & goto :fail )

echo.
echo === Checking dataset ===
if not exist "%DATA%" (
  echo [ERROR] Dataset not found. Put the PaySim file here, then run this again:
  echo   %DATA%
  goto :fail
)
echo Dataset found.

echo.
echo === Backend setup ===
cd /d "%ROOT%backend"
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
  if errorlevel 1 ( echo [ERROR] Could not create venv. & goto :fail )
)
call venv\Scripts\activate.bat
echo Installing backend dependencies...
pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] Backend dependency install failed. & goto :fail )

if not exist .env (
  if exist .env.example copy .env.example .env >nul
  if not exist .env (
    >.env echo MONGO_URI=
    >>.env echo DB_NAME=fraud_detection
    >>.env echo API_KEY=
  )
  echo.
  echo [ACTION NEEDED] backend\.env was just created. Fill in MONGO_URI and API_KEY,
  echo and make frontend\.env VITE_API_KEY match API_KEY, then run setup_all.bat again.
  start notepad "%ROOT%backend\.env"
  goto :fail
)

echo.
echo === Frontend setup ===
cd /d "%ROOT%frontend"
call npm install
if errorlevel 1 ( echo [ERROR] Frontend dependency install failed. & goto :fail )
if not exist .env (
  if exist .env.example copy .env.example .env >nul
  echo [ACTION NEEDED] frontend\.env was just created. Check VITE_API_URL and that
  echo VITE_API_KEY matches backend\.env API_KEY. Then run setup_all.bat again.
  start notepad "%ROOT%frontend\.env"
  goto :fail
)

echo.
echo === Training model (this can take several minutes) ===
cd /d "%ROOT%backend"
python -m app.models.train_model
if errorlevel 1 ( echo [ERROR] Training failed. See the message above. & goto :fail )

echo.
echo === Starting backend and frontend ===
start "Fraud Backend" /d "%ROOT%backend" cmd /k "venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"
start "Fraud Frontend" /d "%ROOT%frontend" cmd /k "npm run dev"
timeout /t 6 >nul
start "" http://localhost:5173
echo.
echo Done. Next time just double-click setup.bat (no retraining).
echo Review the training output above and re-tune the thresholds in
echo backend\app\services\ml_service.py if needed.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
