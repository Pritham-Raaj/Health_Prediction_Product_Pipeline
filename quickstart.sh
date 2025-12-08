#!/bin/bash
# Quick Start Script for Heart Disease ML Service (Linux/Mac/Git Bash)
# Run this to set up everything automatically

set -e

echo "========================================"
echo "Heart Disease ML Service - Quick Start"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.11+ from https://www.python.org/"
    exit 1
fi

echo "[1/6] Creating virtual environment..."
if [ ! -d "ml_venv" ]; then
    python -m venv ml_venv
    echo "Virtual environment created successfully"
else
    echo "Virtual environment already exists"
fi
echo ""

echo "[2/6] Activating virtual environment..."
source ml_venv/Scripts/activate || source ml_venv/bin/activate
echo ""

echo "[3/6] Installing dependencies..."
pip install -r ml_service/requirements.txt
echo ""

echo "[4/6] Checking .env file..."
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please create .env file with your Snowflake credentials"
    echo "See .env.example for template"
    echo ""
    read -p "Press Enter to continue..."
fi
echo ""

echo "[5/6] Checking if model exists..."
if [ ! -f "ml_service/models/heart_disease_model.pkl" ]; then
    echo "Model not found. Training model now..."
    echo "This will take 2-5 minutes..."
    python scripts/train_model_mlflow.py
    echo "Model trained successfully!"
else
    echo "Model already exists"
fi
echo ""

echo "[6/6] Starting ML Service..."
echo ""
echo "========================================"
echo "Service is starting..."
echo "========================================"
echo ""
echo "Access the API at:"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the service"
echo "========================================"
echo ""

cd ml_service
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
