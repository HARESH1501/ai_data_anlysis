@echo off
echo 🚀 Starting Enterprise Analytics Dashboard...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Installing/updating dependencies...
pip install -r requirements.txt --quiet

REM Run the Streamlit dashboard
echo.
echo 🌟 Starting Streamlit dashboard at http://localhost:8501
echo 📊 Enterprise Analytics Dashboard - Power BI Style with ML & AI
echo.
echo Press Ctrl+C to stop the dashboard
echo.

streamlit run app.py

pause