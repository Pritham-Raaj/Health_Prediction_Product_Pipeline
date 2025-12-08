# Heart Disease Prediction ML Service

A production-ready machine learning service for predicting heart disease risk with experiment tracking, model serving, monitoring, drift detection, and explainability.

## Features

- **FastAPI REST API** - High-performance model serving
- **MLflow Integration** - Experiment tracking and model registry
- **SHAP Explainability** - Understand model predictions
- **Evidently AI** - Data drift detection and monitoring
- **Docker Support** - Containerized deployment
- **Render Deployment** - Production-ready cloud deployment
- **Comprehensive Logging** - Structured logging with Loguru
- **Health Checks** - Monitor service health
- **Batch Predictions** - Process multiple predictions efficiently

## Architecture

```
Health_Prediction/
├── ml_service/                 # ML Service application
│   ├── app/
│   │   ├── main.py            # FastAPI application
│   │   ├── model_service.py   # Model inference logic
│   │   └── schemas.py         # Pydantic models
│   ├── config/
│   │   └── settings.py        # Configuration management
│   ├── monitoring/
│   │   └── drift_detector.py  # Drift detection with Evidently
│   ├── models/                # Trained models (generated)
│   │   ├── heart_disease_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_names.json
│   └── requirements.txt       # Python dependencies
├── scripts/
│   ├── train_model_mlflow.py  # Training with MLflow
│   └── ingest_raw.py          # Data ingestion
├── mlflow_tracking/           # MLflow artifacts
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Multi-container setup
└── render.yaml                # Render deployment config
```

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Snowflake account (for data access)
- Git

## Quick Start

### 1. Clone and Setup

```bash
cd Health_Prediction
```

### 2. Configure Environment

Create/update `.env` file with Snowflake credentials:

```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_ANALYTICS_SCHEMA=analytics
SNOWFLAKE_ROLE=your_role
```

### 3. Train Model with MLflow

```bash
# Install dependencies
pip install -r ml_service/requirements.txt

# Train model and save artifacts
python scripts/train_model_mlflow.py
```

This will:
- Load data from Snowflake
- Train baseline models (Logistic Regression, Random Forest, KNN)
- Perform hyperparameter tuning
- Log experiments to MLflow
- Save model artifacts to `ml_service/models/`
- Save reference data for drift detection

### 4. View MLflow UI

```bash
cd mlflow_tracking
mlflow ui
# Open http://localhost:5000
```

### 5. Run Locally (Development)

```bash
# Option A: Direct Python
cd ml_service
python app/main.py

# Option B: Using uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Run with Docker

```bash
# Build and run
docker-compose up --build

# Access services:
# - ML API: http://localhost:8000
# - MLflow UI: http://localhost:5000
# - API Docs: http://localhost:8000/docs
```

### 7. Run with Monitoring Stack (Optional)

```bash
# Include Prometheus and Grafana
docker-compose --profile monitoring up
```

## API Endpoints

### Health Check
```bash
GET /health
```

Returns service health status.

### Single Prediction
```bash
POST /predict
Content-Type: application/json

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

Response:
```json
{
  "prediction": 1,
  "probability": 0.8234,
  "risk_level": "High",
  "model_version": "1.0.0",
  "timestamp": "2025-12-04T10:30:00.000000"
}
```

### Batch Predictions
```bash
POST /predict/batch
Content-Type: application/json

{
  "predictions": [
    { "age": 55, "sex": 1, ... },
    { "age": 48, "sex": 0, ... }
  ]
}
```

### SHAP Explainability
```bash
POST /explain
Content-Type: application/json

{
  "age": 55,
  "sex": 1,
  "chest_pain_type": 2,
  ...
}
```

Response includes SHAP values and top contributing features.

### Drift Detection
```bash
GET /monitoring/drift
```

Returns drift detection results comparing recent predictions to reference data.

### Monitoring Stats
```bash
GET /monitoring/stats
```

Returns monitoring window statistics.

## Deployment to Render

### 1. Prerequisites
- GitHub repository
- Render account
- Model artifacts committed or stored separately

### 2. Push to GitHub
```bash
git add .
git commit -m "Add production ML service"
git push origin main
```

### 3. Deploy to Render

**Option A: Using render.yaml (Recommended)**

1. Connect your GitHub repo to Render
2. Render will automatically detect `render.yaml`
3. Add environment variables in Render dashboard:
   - SNOWFLAKE_ACCOUNT
   - SNOWFLAKE_USER
   - SNOWFLAKE_PASSWORD
   - SNOWFLAKE_WAREHOUSE
   - SNOWFLAKE_DATABASE
   - SNOWFLAKE_ANALYTICS_SCHEMA
   - SNOWFLAKE_ROLE

**Option B: Manual Setup**

1. Create new Web Service on Render
2. Connect GitHub repository
3. Set:
   - Environment: Docker
   - Dockerfile Path: `./Dockerfile`
   - Health Check Path: `/health`
4. Add environment variables
5. Deploy

### 4. Verify Deployment
```bash
curl https://your-app.onrender.com/health
```

## Model Training Pipeline

### Features Used
1. **AGE** - Patient age
2. **SEX_LABEL** - Gender (0=Female, 1=Male)
3. **CHESTPAINTYPE_LABEL** - Chest pain type (0-3)
4. **RESTINGBP_LABEL** - Resting blood pressure category
5. **CHOL** - Cholesterol level
6. **FASTINGBS_LABEL** - Fasting blood sugar
7. **RESTECG_LABEL** - Resting ECG results
8. **MAX_HEARTRATE** - Maximum heart rate achieved
9. **EXERCISE_CHESTPAIN_LABEL** - Exercise-induced angina
10. **OLDPEAK** - ST depression

### Training Process
1. Load data from Snowflake (`analytics.heart_features`)
2. Split into train/test (80/20)
3. Scale features using StandardScaler
4. Train baseline models:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors
5. Select best model based on accuracy
6. Hyperparameter tuning with RandomizedSearchCV
7. Log all experiments to MLflow
8. Save best model artifacts

### Model Performance
- **Model**: Logistic Regression
- **Accuracy**: 83.11%
- **Precision**: 84% (no disease), 82% (disease)
- **Recall**: 85% (no disease), 81% (disease)
- **F1-Score**: 85% (no disease), 81% (disease)

## Monitoring and Drift Detection

### Data Drift
The service automatically monitors predictions for data drift:
- Compares recent predictions against reference data
- Uses Evidently AI for statistical tests
- Detects distribution shifts in features
- Alerts when drift is detected

### Access Drift Report
```bash
curl http://localhost:8000/monitoring/drift
```

### Reference Data
- Saved during model training
- Based on test set
- Location: `ml_service/monitoring/reference_data.csv`
- Update periodically for accurate drift detection

## SHAP Explainability

Each prediction can be explained using SHAP values:
- **Feature Importance**: Which features contributed most
- **Direction**: Positive or negative contribution
- **Magnitude**: Strength of contribution

### Example Explanation
```json
{
  "prediction": 1,
  "probability": 0.8234,
  "shap_values": {
    "AGE": 0.15,
    "SEX_LABEL": 0.12,
    "CHOL": 0.08,
    ...
  },
  "top_features": [
    {"feature": "AGE", "contribution": 0.15},
    {"feature": "SEX_LABEL", "contribution": 0.12},
    ...
  ],
  "base_value": 0.52
}
```

## Testing

### Manual Testing
```bash
# Test health
curl http://localhost:8000/health

# Test prediction
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

### API Documentation
Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

### Model Not Loading
- Check model files exist in `ml_service/models/`
- Run training script: `python scripts/train_model_mlflow.py`
- Check logs: `logs/ml_service.log`

### Snowflake Connection Issues
- Verify `.env` file configuration
- Test connection: `python scripts/train_model_mlflow.py`
- Check Snowflake warehouse is running

### Docker Build Fails
- Ensure model artifacts are present
- Check Docker daemon is running
- Review Dockerfile for errors

### Drift Detection Not Working
- Ensure reference data exists
- Check monitoring window has enough samples (min 10)
- Verify `DRIFT_DETECTION_ENABLED=true`

## Production Considerations

### Security
- Never commit `.env` files
- Use Render environment variables for secrets
- Implement authentication/authorization if needed
- Use HTTPS in production

### Scaling
- Increase Render plan for more resources
- Use load balancer for multiple instances
- Consider async processing for batch predictions
- Cache frequently used data

### Monitoring
- Monitor API response times
- Track prediction latency
- Set up alerts for drift detection
- Monitor resource usage (CPU, memory)

### Model Updates
- Retrain periodically with new data
- Version models in MLflow registry
- Use A/B testing for model rollout
- Monitor model performance metrics

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | Heart Disease Prediction API |
| `APP_VERSION` | Application version | 1.0.0 |
| `DEBUG` | Debug mode | false |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `MODEL_PATH` | Path to model file | ml_service/models/heart_disease_model.pkl |
| `SCALER_PATH` | Path to scaler file | ml_service/models/scaler.pkl |
| `ENABLE_MONITORING` | Enable monitoring | true |
| `DRIFT_DETECTION_ENABLED` | Enable drift detection | true |
| `MONITORING_WINDOW_SIZE` | Max samples in window | 100 |
| `LOG_LEVEL` | Logging level | INFO |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI | mlflow_tracking |

## License

This project is part of the Heart Disease Prediction ML pipeline.

## Support

For issues and questions:
1. Check logs: `logs/ml_service.log`
2. Review API docs: `/docs`
3. Check MLflow experiments
4. Review drift detection reports

## Next Steps

1. ✅ Train model with MLflow tracking
2. ✅ Deploy to local Docker
3. ✅ Test all API endpoints
4. 🚀 Deploy to Render
5. 📊 Set up monitoring dashboards
6. 🔄 Implement CI/CD pipeline
7. 🔐 Add authentication
8. 📈 A/B test model versions
