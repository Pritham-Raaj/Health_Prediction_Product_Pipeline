"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
from enum import Enum

class ChestPainType(str, Enum):
    """Chest pain type categories"""
    TA = "TA"  
    ATA = "ATA" 
    NAP = "NAP" 
    ASY = "ASY" 

class RestECGType(str, Enum):
    """Resting ECG categories"""
    NORMAL = "Normal"
    ST = "ST"
    LVH = "LVH"

class PredictionInput(BaseModel):
    """Input schema for heart disease prediction"""
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    chest_pain_type: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    resting_bp: int = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    cholesterol: int = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dl)")
    fasting_bs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0=No, 1=Yes)")
    resting_ecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    max_heart_rate: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exercise_angina: int = Field(..., ge=0, le=1, description="Exercise induced angina (0=No, 1=Yes)")
    oldpeak: float = Field(..., ge=-5.0, le=10.0, description="ST depression induced by exercise")
    location: int = Field(default=0, ge=0, le=3, description="Location/Site label (0-3)")

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 18 or v > 100:
            raise ValueError('Age should be between 18 and 100 for meaningful predictions')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "sex": 1,
                "chest_pain_type": 2,
                "resting_bp": 140,
                "cholesterol": 250,
                "fasting_bs": 1,
                "resting_ecg": 0,
                "max_heart_rate": 150,
                "exercise_angina": 1,
                "oldpeak": 2.5,
                "location": 0
            }
        }

class PredictionOutput(BaseModel):
    """Output schema for heart disease prediction"""
    prediction: int = Field(..., description="Predicted class (0=No Disease, 1=Disease)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class FeatureContribution(BaseModel):
    """Feature contribution for explainability"""
    feature: str
    contribution: float

class ExplainabilityOutput(BaseModel):
    """Output schema for SHAP explainability"""
    prediction: int
    probability: float
    shap_values: Dict[str, float] = Field(..., description="SHAP values for each feature")
    top_features: List[FeatureContribution] = Field(..., description="Top contributing features")
    base_value: float = Field(..., description="Expected model output")

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    mlflow_connected: bool
    timestamp: str

class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    predictions: List[PredictionInput]

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
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
                        "oldpeak": 2.5,
                        "location": 0
                    }
                ]
            }
        }

class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[PredictionOutput]
    total_predictions: int
    timestamp: str
