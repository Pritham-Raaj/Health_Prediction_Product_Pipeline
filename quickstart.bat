@echo off
REM Quick Start Script for Heart Disease ML Service
REM Run this to set up everything automatically

echo ========================================
echo Heart Disease ML Service - Quick Start
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Creating virtual environment...
if not exist ml_venv (
    python -m venv ml_venv
    echo Virtual environment created successfully
) else (
    echo Virtual environment already exists
)
echo.

echo [2/6] Activating virtual environment...
call ml_venv\Scripts\activate.bat
echo.

echo [3/6] Installing dependencies...
pip install -r ml_service\requirements.txt
echo.

echo [4/6] Checking .env file...
if not exist .env (
    echo WARNING: .env file not found!
    echo Please create .env file with your Snowflake credentials
    echo See .env.example for template
    echo.
    pause
)
echo.

echo [5/6] Checking if model exists...
if not exist ml_service\models\heart_disease_model.pkl (
    echo Model not found. Training model now...
    echo This will take 2-5 minutes...
    python scripts\train_model_mlflow.py
    if errorlevel 1 (
        echo ERROR: Model training failed
        echo Check your Snowflake credentials in .env file
        pause
        exit /b 1
    )
    echo Model trained successfully!
) else (
    echo Model already exists
)
echo.

echo [6/6] Starting ML Service...
echo.
echo ========================================
echo Service is starting...
echo ========================================
echo.
echo Access the API at:
echo   - API Docs: http://localhost:8000/docs
echo   - Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the service
echo ========================================
echo.

cd ml_service
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
