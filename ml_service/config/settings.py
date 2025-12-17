"""
Configuration settings for the Heart Disease Prediction ML Service
"""
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import os
BASE_DIR = Path(__file__).parent.parent.absolute()  

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "Heart Disease Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MODEL_PATH: str = str(BASE_DIR / "models" / "heart_disease_model.pkl")
    SCALER_PATH: str = str(BASE_DIR / "models" / "scaler.pkl")
    FEATURE_NAMES_PATH: str = str(BASE_DIR / "models" / "feature_names.json")
    MLFLOW_TRACKING_URI: str = str(BASE_DIR.parent / "mlflow_tracking")
    MLFLOW_EXPERIMENT_NAME: str = "heart_disease_prediction"
    
    SNOWFLAKE_ACCOUNT: Optional[str] = None
    SNOWFLAKE_USER: Optional[str] = None
    SNOWFLAKE_PASSWORD: Optional[str] = None
    SNOWFLAKE_WAREHOUSE: Optional[str] = None
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_ANALYTICS_SCHEMA: Optional[str] = None
    SNOWFLAKE_ROLE: Optional[str] = None
    
    ENABLE_SHAP: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(BASE_DIR.parent / "logs" / "ml_service.log")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
