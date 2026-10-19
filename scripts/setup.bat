@echo off
setlocal
cd /d "%~dp0.."
echo === setup ===
py -3.12 --version >nul 2>&1
if errorlevel 1 goto nopython
if not exist .venv\Scripts\python.exe py -3.12 -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto fail
python src\hand_tracking\hand_demo.py --selftest
if errorlevel 1 goto fail
echo.
echo Setup OK. Now run scripts\run.bat
pause
exit /b 0

:nopython
echo [ERROR] Python 3.12 was not found.
echo Install Python 3.12 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" and "py launcher" in the installer.
echo Then run setup.bat again.
pause
exit /b 1

:fail
echo [ERROR] Something failed. See the messages above.
pause
exit /b 1
