@echo off
:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python.
    exit /b
)

:: Navigate to the chatbot directory
cd /d "%~dp0"

:: Create a virtual environment (if necessary)
IF NOT EXIST "env" (
    echo Creating virtual environment...
    python -m venv env
)

:: Activate the virtual environment
call env\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Run the chatbot app
start "" streamlit run study_chatbot.py

:: Keep the command window open
pause
