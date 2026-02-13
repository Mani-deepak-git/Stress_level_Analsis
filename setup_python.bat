@echo off
echo ========================================
echo Setting up Python 3.10 Environment
echo ========================================

REM Check if Python 3.10 is installed
python --version | findstr "3.10" >nul
if %errorlevel% neq 0 (
    echo ERROR: Python 3.10.x is required but not found
    echo Please install Python 3.10.x from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python 3.10 detected. Creating virtual environment...

REM Create virtual environment
python -m venv ai_env

REM Activate virtual environment
call ai_env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo Installing ML dependencies...
pip install -r requirements.txt

echo ========================================
echo Python environment setup complete!
echo ========================================
echo To activate: ai_env\Scripts\activate.bat
echo To deactivate: deactivate
pause