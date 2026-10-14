@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto error
pip install -r requirements.txt
if errorlevel 1 goto error
echo.
echo Starting simplified Smriti AI...
python -m streamlit run simple_app.py
if errorlevel 1 goto error
goto end
:error
echo.
echo Setup or launch failed. Read the error above, check Python 3.10+ and your network, then try again.
pause
:end
endlocal
