# VS Code Quick Start Guide - Heart Disease ML Service

Complete guide to running and developing the ML service using Visual Studio Code.

## 📋 Prerequisites

1. **VS Code** installed (download from https://code.visualstudio.com/)
2. **Python 3.11+** installed
3. **Git** installed
4. **Docker Desktop** (optional, for containerization)

## 🚀 Initial Setup (One-Time)

### Step 1: Open Project in VS Code

**Method 1: From VS Code**
1. Open VS Code
2. File → Open Folder
3. Navigate to: `C:\Users\prith\vr_env\Health_Prediction`
4. Click "Select Folder"

**Method 2: From Command Line**
```bash
cd C:\Users\prith\vr_env\Health_Prediction
code .
```

### Step 2: Install Required Extensions

Click Extensions icon (Ctrl+Shift+X) and install:

**Essential:**
- ✅ Python (by Microsoft)
- ✅ Pylance (by Microsoft)
- ✅ Python Debugger (by Microsoft)

**Recommended:**
- Docker (by Microsoft)
- Thunder Client (for API testing)
- GitLens
- Better Comments
- Error Lens

### Step 3: Create Virtual Environment

**Open integrated terminal**: Press `` Ctrl+` `` or View → Terminal

```bash
# Create virtual environment
python -m venv ml_venv

# Activate it (choose based on your terminal)

# PowerShell:
.\ml_venv\Scripts\Activate.ps1

# Command Prompt:
ml_venv\Scripts\activate.bat

# Git Bash:
source ml_venv/Scripts/activate
```

You should see `(ml_venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
# Install all required packages
pip install -r ml_service/requirements.txt

# Verify installation
pip list
```

### Step 5: Select Python Interpreter

1. Press `Ctrl+Shift+P` (Command Palette)
2. Type: "Python: Select Interpreter"
3. Choose: `Python 3.11.x ('ml_venv': venv)`

You'll see the interpreter in the bottom-right corner of VS Code.

### Step 6: Trust the Workspace

If prompted, click "Yes, I trust the authors" to enable all features.

---

## 🎯 Running the ML Service

### Method 1: Using VS Code Debugger (Recommended)

**Easiest way to run and debug:**

1. Click Run & Debug icon (Ctrl+Shift+D) or press F5
2. Select: **"Run ML Service (FastAPI)"**
3. Click green play button or press F5

The service will start with auto-reload enabled!

**What you'll see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Access the API:**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Method 2: Using Terminal

```bash
# Navigate to ml_service directory
cd ml_service

# Run with uvicorn
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Method 3: Using VS Code Tasks

1. Press `Ctrl+Shift+P`
2. Type: "Tasks: Run Task"
3. Select: **"Start ML Service"**

### Method 4: Simple Python Run

```bash
cd ml_service
python app/main.py
```

---

## 🧪 Training the Model

### Before Running the Service

You need to train the model first to create the required artifacts.

### Method 1: Using Debugger

1. Press F5 or click Run & Debug
2. Select: **"Train Model with MLflow"**
3. Click play button

### Method 2: Using Terminal

```bash
# Make sure you're in the project root
cd C:\Users\prith\vr_env\Health_Prediction

# Ensure virtual environment is activated
# (You should see (ml_venv) in prompt)

# Run training script
python scripts/train_model_mlflow.py
```

### Method 3: Using Tasks

1. `Ctrl+Shift+P` → "Tasks: Run Task"
2. Select: **"Train Model"**

**What happens:**
1. Connects to Snowflake
2. Loads training data
3. Trains 3 models (Logistic Regression, Random Forest, KNN)
4. Tunes hyperparameters
5. Logs to MLflow
6. Saves models to `ml_service/models/`
7. Creates reference data for drift detection

**Expected time:** 2-5 minutes

**Output files created:**
- `ml_service/models/heart_disease_model.pkl`
- `ml_service/models/scaler.pkl`
- `ml_service/models/feature_names.json`
- `ml_service/models/model_metadata.json`
- `ml_service/monitoring/reference_data.csv`

---

## 📊 Viewing MLflow Experiments

### Method 1: Using Tasks

1. `Ctrl+Shift+P` → "Tasks: Run Task"
2. Select: **"Start MLflow UI"**
3. Open: http://localhost:5000

### Method 2: Using Terminal

```bash
mlflow ui --backend-store-uri mlflow_tracking
```

Then open http://localhost:5000 in your browser.

**What you can do:**
- View all experiment runs
- Compare model performance
- Check hyperparameters
- Download model artifacts
- See training metrics graphs

---

## 🧪 Testing the API

### Method 1: Using Interactive Docs (Easiest)

1. Start the ML service (F5)
2. Open: http://localhost:8000/docs
3. Try the endpoints:
   - Click on any endpoint (e.g., `/predict`)
   - Click "Try it out"
   - Fill in the example data
   - Click "Execute"
   - See the response!

### Method 2: Using Thunder Client (VS Code Extension)

1. Install "Thunder Client" extension
2. Click Thunder Client icon
3. New Request
4. Method: POST
5. URL: `http://localhost:8000/predict`
6. Body (JSON):
```json
{
  "age": 55,
  "sex": 1,
  "chest_pain_type": 2,
  "resting_bp": 140,
  "cholesterol": 250,
  "fasting_bs": 1,
  "resting_ecg": 0,
  "max_heart_rate": 150,
  "exercise_angina": 1,
  "oldpeak": 2.5
}
```
7. Send!

### Method 3: Using Terminal (curl)

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\": 55, \"sex\": 1, \"chest_pain_type\": 2, \"resting_bp\": 140, \"cholesterol\": 250, \"fasting_bs\": 1, \"resting_ecg\": 0, \"max_heart_rate\": 150, \"exercise_angina\": 1, \"oldpeak\": 2.5}"

# Check drift
curl http://localhost:8000/monitoring/drift

# Get explanation
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d "{\"age\": 55, \"sex\": 1, \"chest_pain_type\": 2, \"resting_bp\": 140, \"cholesterol\": 250, \"fasting_bs\": 1, \"resting_ecg\": 0, \"max_heart_rate\": 150, \"exercise_angina\": 1, \"oldpeak\": 2.5}"
```

### Method 4: Using Python (in VS Code Terminal)

Create a test file `test_api_manual.py`:

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print("Health:", response.json())

# Make prediction
payload = {
    "age": 55,
    "sex": 1,
    "chest_pain_type": 2,
    "resting_bp": 140,
    "cholesterol": 250,
    "fasting_bs": 1,
    "resting_ecg": 0,
    "max_heart_rate": 150,
    "exercise_angina": 1,
    "oldpeak": 2.5
}

response = requests.post("http://localhost:8000/predict", json=payload)
print("Prediction:", response.json())
```

Run it:
```bash
python test_api_manual.py
```

---

## 🧪 Running Tests

### Method 1: Using Test Explorer (Best)

1. Click Testing icon in left sidebar (flask icon)
2. Click "Configure Python Tests"
3. Select "pytest"
4. Select "tests" directory
5. Tests will appear in the sidebar
6. Click play button next to any test or "Run All Tests"

### Method 2: Using Debugger

1. F5 → Select: **"Run Tests (pytest)"**
2. View results in terminal

### Method 3: Using Terminal

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=ml_service --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::TestHealthEndpoints::test_health_check -v
```

### Method 4: Using Tasks

1. `Ctrl+Shift+P` → "Tasks: Run Task"
2. Select: **"Run Tests"**

---

## 🐳 Using Docker

### Start with Docker Compose

**Method 1: Using Tasks**
1. `Ctrl+Shift+P` → "Tasks: Run Task"
2. Select: **"Docker Compose Up"**

**Method 2: Using Terminal**
```bash
docker-compose up --build
```

**Method 3: Using Docker Extension**
1. Install Docker extension
2. Right-click `docker-compose.yml`
3. Select "Compose Up"

**Services started:**
- ML Service: http://localhost:8000
- MLflow UI: http://localhost:5000

### Stop Docker Compose

**Method 1: Using Tasks**
1. `Ctrl+Shift+P` → "Tasks: Run Task"
2. Select: **"Docker Compose Down"**

**Method 2: Using Terminal**
```bash
docker-compose down
```

**Method 3: Press Ctrl+C in the terminal where it's running**

---

## 🔧 Debugging

### Setting Breakpoints

1. Click in the left margin (gutter) next to line numbers
2. Red dot appears = breakpoint set
3. Run with debugger (F5)
4. Code will pause at breakpoint
5. Use debug toolbar:
   - Continue (F5)
   - Step Over (F10)
   - Step Into (F11)
   - Step Out (Shift+F11)

### Debug Variables

When paused at breakpoint:
- **Variables panel**: See all variables and their values
- **Watch panel**: Add expressions to monitor
- **Call Stack**: See function call hierarchy
- **Debug Console**: Execute code at breakpoint

### Debug API Requests

1. Set breakpoint in `ml_service/app/main.py` in the `/predict` endpoint
2. Start debugger (F5)
3. Make API request (using browser, Thunder Client, or curl)
4. Debugger will pause
5. Inspect `input_data`, `prediction`, etc.

---

## 📝 Common Tasks Quick Reference

### Quick Commands (Keyboard Shortcuts)

| Action | Shortcut |
|--------|----------|
| Open Command Palette | `Ctrl+Shift+P` |
| Open Terminal | `` Ctrl+` `` |
| Start Debugging | `F5` |
| Toggle Breakpoint | `F9` |
| Run Tests | Testing sidebar |
| Search Files | `Ctrl+P` |
| Search in Files | `Ctrl+Shift+F` |
| Go to Definition | `F12` |
| Find All References | `Shift+F12` |
| Format Document | `Shift+Alt+F` |

### Daily Workflow

**Start of Day:**
```bash
# 1. Open VS Code
code C:\Users\prith\vr_env\Health_Prediction

# 2. Activate environment (if not auto-activated)
# Terminal will show (ml_venv)

# 3. Pull latest changes (if using Git)
git pull

# 4. Start ML service
Press F5 → Select "Run ML Service (FastAPI)"
```

**During Development:**
1. Make code changes
2. Save file (Ctrl+S) → Auto-reload happens
3. Test changes at http://localhost:8000/docs
4. Set breakpoints and debug as needed
5. Run tests frequently

**End of Day:**
```bash
# Stop the service
Ctrl+C in terminal or stop debugger

# Commit changes (if ready)
git add .
git commit -m "Your message"
git push
```

---

## 📂 Project Structure in VS Code

```
Health_Prediction/
├── .vscode/                  # VS Code configuration
│   ├── settings.json        # Workspace settings
│   ├── launch.json          # Debug configurations
│   └── tasks.json           # Task definitions
├── ml_service/              # Main service
│   ├── app/
│   │   ├── main.py         # FastAPI app (edit this for new endpoints)
│   │   ├── model_service.py # Model logic
│   │   └── schemas.py      # Pydantic models
│   ├── config/
│   │   └── settings.py     # Configuration
│   ├── monitoring/
│   │   └── drift_detector.py
│   ├── models/             # Generated model files
│   └── requirements.txt    # Dependencies
├── scripts/
│   └── train_model_mlflow.py # Training script
├── tests/
│   └── test_api.py         # API tests
├── mlflow_tracking/        # MLflow artifacts
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-container setup
└── .env                    # Environment variables (Snowflake creds)
```

---

## 🐛 Troubleshooting

### "Module not found" errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r ml_service/requirements.txt

# OR recreate virtual environment
rm -rf ml_venv
python -m venv ml_venv
ml_venv\Scripts\activate
pip install -r ml_service/requirements.txt
```

### "Model files not found"

**Solution:**
```bash
# Train the model first
python scripts/train_model_mlflow.py
```

### "Port 8000 already in use"

**Solution:**
```bash
# Find and kill process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# OR use different port in launch.json
Change "--port", "8000" to "--port", "8001"
```

### Debugger not working

**Solutions:**
1. Ensure virtual environment is selected (bottom-right corner)
2. Reload VS Code window: `Ctrl+Shift+P` → "Reload Window"
3. Check `launch.json` paths are correct
4. Install Python Debugger extension

### Tests not appearing

**Solutions:**
1. Click Testing icon → Configure Tests
2. Select pytest
3. Select tests folder
4. Reload window if needed

### Snowflake connection issues

**Solution:**
Check `.env` file has correct credentials:
```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
# ... etc
```

---

## 💡 Pro Tips

### 1. Use Multi-cursor Editing
- `Alt+Click` to add cursors
- `Ctrl+D` to select next occurrence
- `Ctrl+Shift+L` to select all occurrences

### 2. Quick File Navigation
- `Ctrl+P` → Type filename → Enter
- `Ctrl+Tab` to switch between open files

### 3. Code Formatting
- `Shift+Alt+F` to format current file
- Set "Format On Save" in settings

### 4. Git Integration
- View changes: Source Control icon (Ctrl+Shift+G)
- Stage files, commit, push directly from VS Code
- Install GitLens for more features

### 5. Terminal Splits
- `Ctrl+Shift+5` to split terminal
- Run ML service in one, tests in another

### 6. Workspace Snippets
Create custom snippets for common code patterns:
- File → Preferences → User Snippets

---

## 📚 Useful Extensions for This Project

**Development:**
- Python Test Explorer
- autoDocstring (for documentation)
- Python Type Hint

**Docker:**
- Docker
- Remote - Containers

**API Development:**
- Thunder Client (lightweight API client)
- REST Client

**Productivity:**
- Todo Tree (track TODOs in code)
- Better Comments
- Bracket Pair Colorizer

---

## ✅ Checklist: First Time Setup

- [ ] VS Code installed
- [ ] Python extension installed
- [ ] Project opened in VS Code
- [ ] Virtual environment created (`ml_venv`)
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Python interpreter selected in VS Code
- [ ] `.env` file configured with Snowflake credentials
- [ ] Model trained (ran `train_model_mlflow.py`)
- [ ] Service runs successfully (F5)
- [ ] Can access http://localhost:8000/docs
- [ ] Tests run successfully
- [ ] Debugger works (set breakpoint and hit it)

---

## 🎓 Learning Resources

**VS Code Python:**
- https://code.visualstudio.com/docs/python/python-tutorial

**FastAPI:**
- https://fastapi.tiangolo.com/tutorial/

**Python Debugging in VS Code:**
- https://code.visualstudio.com/docs/python/debugging

---

**You're all set! Press F5 to start developing! 🚀**
