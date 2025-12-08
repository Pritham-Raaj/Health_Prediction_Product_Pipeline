# Heart Disease Prediction ML Service - Deployment Guide

Complete guide for deploying the production ML service from development to Render.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Training the Model](#training-the-model)
4. [Testing Locally](#testing-locally)
5. [Docker Deployment](#docker-deployment)
6. [Render Cloud Deployment](#render-cloud-deployment)
7. [Post-Deployment](#post-deployment)
8. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### Required Software
- **Python** 3.11+
- **Docker** 20.10+ and Docker Compose
- **Git**
- **Snowflake** account with data access
- **Render** account (free tier works)

### Required Accounts
- Snowflake account with `analytics.heart_features` table
- GitHub account
- Render account (sign up at https://render.com)

### Environment Setup
Ensure you have:
- `.env` file configured with Snowflake credentials
- Access to the Health_Prediction repository
- Conda or venv for Python environments

---

## Local Development Setup

### 1. Clone/Navigate to Project
```bash
cd C:\Users\prith\vr_env\Health_Prediction
```

### 2. Create Environment File
Create `.env` if it doesn't exist:
```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_ANALYTICS_SCHEMA=analytics
SNOWFLAKE_ROLE=your_role
```

### 3. Install Dependencies
```bash
# Install ML service dependencies
pip install -r ml_service/requirements.txt

# Or use virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r ml_service/requirements.txt
```

---

## Training the Model

### Step 1: Prepare Data with DBT
```bash
cd health_dbt
source ../../dbt_venv/Scripts/activate
dbt run
dbt test
```

This creates/updates:
- `analytics.heart_curated` - Cleaned data
- `analytics.heart_features` - ML-ready features

### Step 2: Train Model with MLflow
```bash
cd ..
python scripts/train_model_mlflow.py
```

**What this does:**
1. Connects to Snowflake
2. Loads `analytics.heart_features`
3. Trains 3 baseline models (Logistic Regression, Random Forest, KNN)
4. Selects best model
5. Hyperparameter tuning with RandomizedSearchCV
6. Logs everything to MLflow
7. Saves model artifacts to `ml_service/models/`:
   - `heart_disease_model.pkl` - Trained model
   - `scaler.pkl` - Feature scaler
   - `feature_names.json` - Feature list
   - `model_metadata.json` - Metadata
8. Saves reference data for drift detection

**Expected Output:**
```
=================================================
Heart Disease Prediction Model Training
=================================================
...
Best Model: Logistic Regression
Accuracy: 0.8311
Precision: 0.8400
Recall: 0.8500
F1 Score: 0.8500
ROC-AUC: 0.8932
```

### Step 3: View MLflow Experiments
```bash
cd mlflow_tracking
mlflow ui
```
Open: http://localhost:5000

Review:
- All experiment runs
- Model parameters
- Metrics comparison
- Model artifacts

---

## Testing Locally

### Option 1: Direct Python
```bash
cd ml_service
python app/main.py
```

### Option 2: Uvicorn (with auto-reload)
```bash
cd ml_service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Points
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Test Endpoints

**1. Health Check**
```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "mlflow_connected": true,
  "timestamp": "2025-12-04T10:00:00.000000"
}
```

**2. Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Expected:
```json
{
  "prediction": 1,
  "probability": 0.8234,
  "risk_level": "High",
  "model_version": "1.0.0",
  "timestamp": "2025-12-04T10:00:00.000000"
}
```

**3. SHAP Explanation**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{ ... same as above ... }'
```

**4. Drift Check**
```bash
curl http://localhost:8000/monitoring/drift
```

---

## Docker Deployment

### Step 1: Ensure Model Artifacts Exist
```bash
ls ml_service/models/
# Should show:
# heart_disease_model.pkl
# scaler.pkl
# feature_names.json
# model_metadata.json
```

### Step 2: Build Docker Image
```bash
docker build -t heart-disease-ml-service .
```

### Step 3: Test Docker Image
```bash
docker run -d -p 8000:8000 \
  --name test-ml-service \
  -v $(pwd)/ml_service/models:/app/models \
  heart-disease-ml-service

# Wait 10 seconds for startup
sleep 10

# Test health
curl http://localhost:8000/health

# Cleanup
docker stop test-ml-service
docker rm test-ml-service
```

### Step 4: Run with Docker Compose
```bash
docker-compose up --build
```

This starts:
- **ML Service** on port 8000
- **MLflow Server** on port 5000

Optional: Add monitoring
```bash
docker-compose --profile monitoring up
```

This adds:
- **Prometheus** on port 9090
- **Grafana** on port 3000 (admin/admin)

---

## Render Cloud Deployment

### Prerequisites
1. GitHub repository with code
2. Render account
3. Model artifacts committed or built during deployment

### Step 1: Prepare Repository

**Option A: Commit Model Artifacts (Easier)**
```bash
git add ml_service/models/*.pkl ml_service/models/*.json
git commit -m "Add trained model artifacts"
git push origin main
```

**Option B: Build During Deployment**
- Requires Snowflake credentials in Render
- Training runs during deployment
- Slower but always uses latest data

### Step 2: Connect GitHub to Render

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub account
4. Select `Health_Prediction` repository

### Step 3: Configure Web Service

**Basic Settings:**
- **Name**: `heart-disease-ml-api`
- **Region**: Oregon (or closest to you)
- **Branch**: `main`
- **Root Directory**: (leave blank)
- **Environment**: Docker
- **Dockerfile Path**: `./Dockerfile`
- **Docker Context**: `.`

**Instance Type:**
- Free tier (512MB RAM) - OK for testing
- Starter ($7/month, 1GB RAM) - Recommended for production

### Step 4: Set Environment Variables

In Render dashboard, add:

```
APP_NAME=Heart Disease Prediction API
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO
PORT=8000
ENABLE_MONITORING=true
DRIFT_DETECTION_ENABLED=true
MLFLOW_TRACKING_URI=/app/mlflow_tracking
```

Optional (if training on deployment):
```
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_ANALYTICS_SCHEMA=analytics
SNOWFLAKE_ROLE=your_role
```

### Step 5: Configure Health Check

- **Health Check Path**: `/health`
- **Health Check Delay**: 40 seconds

### Step 6: Deploy

1. Click "Create Web Service"
2. Render will:
   - Pull code from GitHub
   - Build Docker image
   - Deploy container
   - Run health checks
3. Wait 5-10 minutes for initial deployment

### Step 7: Verify Deployment

Your service URL: `https://heart-disease-ml-api.onrender.com`

Test:
```bash
# Health check
curl https://heart-disease-ml-api.onrender.com/health

# API docs
open https://heart-disease-ml-api.onrender.com/docs

# Prediction
curl -X POST https://heart-disease-ml-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

## Post-Deployment

### 1. Set Up Auto-Deploy

Render automatically deploys on push to main:
```bash
git add .
git commit -m "Update model"
git push origin main
# Render auto-deploys
```

### 2. Configure Custom Domain (Optional)

In Render dashboard:
1. Go to service settings
2. Add custom domain
3. Update DNS records
4. Enable HTTPS

### 3. Set Up Monitoring

**Render Built-in:**
- Metrics: CPU, memory, response times
- Logs: Real-time log streaming
- Alerts: Email notifications

**External (Recommended):**
- Use Prometheus + Grafana
- Set up Sentry for error tracking
- Configure Datadog/New Relic

### 4. API Key/Authentication (Recommended)

Add API key authentication:
```python
# In ml_service/app/main.py
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict(
    input_data: PredictionInput,
    api_key: str = Depends(api_key_header)
):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of code
```

---

## Monitoring & Maintenance

### Daily Checks
- [ ] Health endpoint responds
- [ ] Check error logs
- [ ] Review response times

### Weekly Checks
- [ ] Check drift detection reports
- [ ] Review prediction distribution
- [ ] Monitor resource usage
- [ ] Check for failed predictions

### Monthly Maintenance
- [ ] Retrain model with latest data
- [ ] Update reference data
- [ ] Review and update dependencies
- [ ] Analyze model performance metrics
- [ ] Test all API endpoints
- [ ] Review security patches

### Model Retraining Workflow
```bash
# 1. Pull latest code
git pull origin main

# 2. Update DBT models
cd health_dbt
dbt run

# 3. Retrain model
cd ..
python scripts/train_model_mlflow.py

# 4. Review MLflow metrics
mlflow ui

# 5. Test locally
cd ml_service
python app/main.py
# Test endpoints

# 6. Deploy
git add ml_service/models/
git commit -m "Update model - [YYYY-MM-DD]"
git push origin main
```

### Monitoring Drift

```bash
# Check drift via API
curl https://your-app.onrender.com/monitoring/drift

# If drift detected:
# 1. Investigate features with drift
# 2. Check data quality
# 3. Consider retraining
# 4. Update reference data
```

### Troubleshooting Production Issues

**Service Down:**
1. Check Render dashboard for errors
2. View logs: Render dashboard → Logs
3. Check health endpoint
4. Verify environment variables
5. Review recent deployments

**Slow Responses:**
1. Check resource usage
2. Consider upgrading instance
3. Implement caching
4. Optimize model inference
5. Use batch predictions

**High Error Rate:**
1. Review error logs
2. Check input validation
3. Verify model loading
4. Test with known good inputs
5. Roll back if needed

---

## Rollback Procedure

### Quick Rollback (Render)
1. Go to Render dashboard
2. Select service
3. Go to "Deploys" tab
4. Find last good deploy
5. Click "Redeploy"

### Git Rollback
```bash
# Find last good commit
git log --oneline

# Revert to commit
git revert <commit-hash>
git push origin main
```

---

## Cost Optimization

### Render Pricing
- **Free Tier**: 750 hours/month, 512MB RAM, sleeps after inactivity
- **Starter**: $7/month, 1GB RAM, always on
- **Standard**: $25/month, 2GB RAM, better performance

### Tips to Reduce Costs
1. Use free tier for development/testing
2. Implement request caching
3. Use batch predictions
4. Optimize Docker image size
5. Monitor and reduce unnecessary logging

---

## Security Checklist

- [ ] Environment variables in Render (not in code)
- [ ] HTTPS enabled
- [ ] API authentication implemented
- [ ] Rate limiting configured
- [ ] Input validation with Pydantic
- [ ] Dependencies up to date
- [ ] No secrets in logs
- [ ] CORS properly configured
- [ ] Regular security audits

---

## Additional Resources

- **API Documentation**: `/docs` endpoint
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **Render Documentation**: https://render.com/docs
- **Evidently AI Documentation**: https://docs.evidentlyai.com

---

## Support

For issues:
1. Check logs in Render dashboard
2. Review `ML_SERVICE_README.md`
3. Check GitHub Issues
4. Review MLflow experiments
5. Test locally first

---

## Deployment Checklist

Use this before each deployment:

- [ ] Code changes tested locally
- [ ] Model retrained if needed
- [ ] MLflow experiments reviewed
- [ ] Tests passing: `pytest tests/ -v`
- [ ] Docker build successful
- [ ] Health endpoint works
- [ ] All API endpoints tested
- [ ] Environment variables verified
- [ ] Documentation updated
- [ ] Drift detection working
- [ ] Committed to GitHub
- [ ] Deployment successful on Render
- [ ] Production endpoint tested
- [ ] Monitoring configured
- [ ] Logs reviewed

---

**Last Updated**: 2025-12-04
**Version**: 1.0.0
