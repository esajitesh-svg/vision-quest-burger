@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto failed
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python -m streamlit run simple_app.py
goto end
:failed
echo Setup or launch failed. Check Python 3.10+ and the error above.
pause
:end
endlocal
