# Heart Disease Prediction: End-to-End ML Pipeline

A production-ready machine learning system that demonstrates modern data engineering and MLOps practices through heart disease risk prediction.

# Project Overview

This project demonstrates a complete ML lifecycle:

1. ETL Pipeline - DBT + Snowflake for data transformation
2. Model Training - Scikit-learn with MLflow experiment tracking for developing models locally
3. Production Deployment - FastAPI REST API with Docker containerization

# Key Technologies

| Component | Technology |
| Data Warehouse | Snowflake |
| ETL/Transform | DBT (Data Build Tool) |
| ML Framework | Scikit-learn |
| Experiment Tracking | MLflow (locally) |
| API Framework | FastAPI |
| Containerization | Docker |
| Production | Render (Cloud Platform) |

---

# Architecture

```
┌─────────────────┐
│   Snowflake     │  ← Raw Data Storage
│  (Data Warehouse)│
└────────┬────────┘
         │
    ┌────▼────┐
    │   DBT   │  ← ETL: Transform & Feature Engineering
    └────┬────┘
         │
    ┌────▼────────────┐
    │ ML Training     │  ← Model Training with MLflow(locally)
    │ (scikit-learn)  │
    └────┬────────────┘
         │
    ┌────▼──────────┐
    │ Model Artifacts│  ← Saved models (pkl files)
    └────┬──────────┘
         │
    ┌────▼────────┐
    │   FastAPI   │  ← REST API for predictions
    └────┬────────┘
         │
    ┌────▼────────┐
    │   Docker    │  ← Containerized deployment
    └────┬────────┘
         │
    ┌────▼────────┐
    │   Render    │  ← Cloud hosting
    └─────────────┘
```

---

# Quick Start

# Prerequisites

- Python 3.11+
- Snowflake account (for ETL pipeline)
- Docker (optional, for containerization)

# 1. Clone and Setup

```bash
git clone <repository-url>
cd Health_Prediction
```

# 2. Run ETL Pipeline (DBT + Snowflake)

```bash
# Navigate to DBT project
cd health_dbt

# Setup environment variables
cp .env.example .env
# Edit .env with your Snowflake credentials

# Run DBT transformations
dbt deps
dbt run
dbt test
```

This creates:
- `analytics.heart_curated` - Cleaned dataset (739 records)
- `analytics.heart_features` - Engineered features for ML

# 3. Train ML Model

```bash
# From project root
cd ..

# Install dependencies
pip install -r ml_service/requirements.txt

# Train model with MLflow tracking
python scripts/train_model_mlflow.py
```

Output:
- Model artifacts saved to `ml_service/models/`
- Best model: Logistic Regression (80.41% accuracy)

# 4. Start ML Service

```bash
cd ml_service
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

OR: Docker
```bash
docker-compose up --build
```

# 5. Test the API

Access the interactive documentation:
- Swagger UI: https://health-prediction-8l4d.onrender.com/docs
- Health Check: https://health-prediction-8l4d.onrender.com/health



# Project Structure

```
Health_Prediction/
├── health_dbt/              # DBT project for ETL
│   ├── models/
│   │   ├── curated/         # Cleaned data models
│   │   └── features/        # Feature engineering
│   ├── dbt_project.yml
│   └── profiles.yml
├── ml_service/              # Production ML Service
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   ├── model_service.py # Model inference logic
│   │   └── schemas.py       # API request/response models
│   ├── config/
│   │   └── settings.py      # Configuration management
│   ├── models/              # Trained model artifacts
│   │   ├── heart_disease_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_names.json
│   └── requirements.txt
├── scripts/
│   └── train_model_mlflow.py  # Model training script
├── docker-compose.yml         # Docker orchestration
├── Dockerfile                 # Container definition
└── README.md
```

---

# Technical Highlights

# 1. ETL Pipeline (DBT + Snowflake)

Data Transformation:
```sql
-- Example: heart_curated.sql
WITH cleaned_data AS (
    SELECT
        AGE,
        SEX,
        LOCATION,
        CHOL,
        RESTINGBP,
        ...
    FROM {{ source('raw', 'heart_data') }}
    WHERE AGE > 0
      AND CHOL > 0
)
SELECT * FROM cleaned_data
```

Benefits:
- Version-controlled SQL transformations and access control
- Data quality tests
- Incremental processing
- Documentation generation

# 2. Model Training with MLflow

Tracks(locally):
- Model hyperparameters
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Training artifacts
- Model versioning


# 3. Production API (FastAPI)

Core Endpoints:
- `POST /predict` - Single patient prediction
- `POST /predict/batch` - Multiple predictions
- `POST /explain` - SHAP-based model explanations
- `GET /health` - Service health check

Features:
- Automatic API documentation
- Error handling
- Structured logging
- Health monitoring

# 4. Containerization & Deployment

Docker Features:
- Multi-stage builds for optimized size
- Health checks
- Volume mounts for persistence
- Environment-based configuration

---

# Model Performance

| Metric | Value |
| Algorithm | Logistic Regression |
| Accuracy | 80.41% |
| Precision | 81.58% |
| Recall | 80.52% |
| F1 Score | 81.05% |
| ROC-AUC | 88.95% |

Dataset:
- 739 patient records
- 11 features (age, sex, cholesterol, blood pressure, etc.)
- Binary classification (heart disease: yes/no)
- Balanced classes (51.7% positive)

---

# API Examples

# Health Check
```bash
curl https://health-prediction-8l4d.onrender.com/health
```

# Prediction (High Risk)
```json
{
  "age": 65,
  "sex": 1,
  "chest_pain_type": 3,
  "resting_bp": 160,
  "cholesterol": 280,
  "fasting_bs": 1,
  "resting_ecg": 1,
  "max_heart_rate": 120,
  "exercise_angina": 1,
  "oldpeak": 3.0,
  "location": 0
}
```

# Prediction (Low Risk)
```json
{
  "age": 30,
  "sex": 0,
  "chest_pain_type": 0,
  "resting_bp": 110,
  "cholesterol": 160,
  "fasting_bs": 0,
  "resting_ecg": 0,
  "max_heart_rate": 180,
  "exercise_angina": 0,
  "oldpeak": 0.0,
  "location": 0
}
```

---

# Documentation

- **API Docs**: https://health-prediction-8l4d.onrender.com/docs (Swagger UI)
- **DBT Docs**: Run `dbt docs generate && dbt docs serve`

---




# Acknowledgments

- Dataset: UCI Machine Learning Repository
- Frameworks: FastAPI, MLflow, DBT
- Cloud: Snowflake, Render
