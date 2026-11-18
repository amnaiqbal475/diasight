@echo off
rem DiaSight: double-click to start. The first run sets everything up (about 5-10 minutes, needs internet).
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto start

echo.
echo  First run: creating the Python environment and installing packages.
echo  This takes 5-10 minutes and only happens once.
echo.
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo  Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
  echo  and tick "Add python.exe to PATH" during setup. Then run this file again.
  pause
  exit /b 1
)
%PY% -m venv .venv || goto setupfail
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto setupfail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto setupfail

:start
if not exist "models\best.pt" (
  echo.
  echo  NOTE: models\best.pt was not found, so the app runs in DEMO mode ^(untrained, results mean nothing^).
  echo  Ask for the trained model file and put it in the models folder.
  echo.
)
echo  Starting DiaSight at http://localhost:8501  ^(close this window to stop it^)
rem Open the browser after a short delay; headless mode skips Streamlit's first-run email prompt.
start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app\app.py --server.headless true
pause
exit /b 0

:setupfail
echo.
echo  Setup failed. Check your internet connection, then delete the .venv folder and run this file again.
pause
exit /b 1
