"""
Configuration settings for the Heart Disease Prediction ML Service
"""
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import os


# Get the project root directory (parent of ml_service)
# This file is in ml_service/config/settings.py
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
ML_SERVICE_DIR = BASE_DIR / "ml_service"


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "Heart Disease Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model - Use absolute paths
    MODEL_PATH: str = str(ML_SERVICE_DIR / "models" / "heart_disease_model.pkl")
    SCALER_PATH: str = str(ML_SERVICE_DIR / "models" / "scaler.pkl")
    FEATURE_NAMES_PATH: str = str(ML_SERVICE_DIR / "models" / "feature_names.json")

    # MLflow - Use absolute path
    MLFLOW_TRACKING_URI: str = str(BASE_DIR / "mlflow_tracking")
    MLFLOW_EXPERIMENT_NAME: str = "heart_disease_prediction"

    # Snowflake (optional for production - can load reference data)
    SNOWFLAKE_ACCOUNT: Optional[str] = None
    SNOWFLAKE_USER: Optional[str] = None
    SNOWFLAKE_PASSWORD: Optional[str] = None
    SNOWFLAKE_WAREHOUSE: Optional[str] = None
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_ANALYTICS_SCHEMA: Optional[str] = None
    SNOWFLAKE_ROLE: Optional[str] = None

    # Explainability (OPTIONAL - can be disabled for simpler setup)
    ENABLE_SHAP: bool = True  # Set to False to disable SHAP explanations

    # Logging - Use absolute path
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(BASE_DIR / "logs" / "ml_service.log")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
