"""
FastAPI application for Heart Disease Prediction ML Service
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import (
    PredictionInput,
    PredictionOutput,
    ExplainabilityOutput,
    HealthCheck,
    BatchPredictionInput,
    BatchPredictionOutput
)
from app.model_service import model_service
from config.settings import settings

# Configure logger
logger.add(
    settings.LOG_FILE,
    rotation="10 MB",
    retention="10 days",
    level=settings.LOG_LEVEL
)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready ML API for Heart Disease Prediction with monitoring, drift detection, and explainability",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    if model_service.is_ready():
        logger.info("Model service is ready")
    else:
        logger.warning("Model service is not ready - check model files")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns system health status including model and MLflow connectivity
    """
    try:
        model_loaded = model_service.is_ready()

        # Check MLflow connectivity (simplified)
        mlflow_connected = Path(settings.MLFLOW_TRACKING_URI).exists()

        return HealthCheck(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            mlflow_connected=mlflow_connected,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(input_data: PredictionInput):
    """
    Make a single prediction for heart disease

    - **age**: Patient age in years
    - **sex**: Gender (0=Female, 1=Male)
    - **chest_pain_type**: Type of chest pain (0-3)
    - **resting_bp**: Resting blood pressure (mm Hg)
    - **cholesterol**: Serum cholesterol (mg/dl)
    - **fasting_bs**: Fasting blood sugar > 120 mg/dl (0=No, 1=Yes)
    - **resting_ecg**: Resting ECG results (0-2)
    - **max_heart_rate**: Maximum heart rate achieved
    - **exercise_angina**: Exercise induced angina (0=No, 1=Yes)
    - **oldpeak**: ST depression induced by exercise

    Returns prediction with probability and risk level
    """
    try:
        if not model_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model service is not ready"
            )

        # Make prediction
        prediction, probability = model_service.predict(input_data.dict())

        # Determine risk level
        risk_level = model_service.get_risk_level(probability)

        logger.info(f"Prediction made: {prediction} (probability: {probability:.4f})")

        return PredictionOutput(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            model_version=model_service.model_version,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Make predictions for multiple patients

    Accepts a list of patient data and returns predictions for all
    """
    try:
        if not model_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model service is not ready"
            )

        predictions = []

        for input_data in batch_input.predictions:
            prediction, probability = model_service.predict(input_data.dict())
            risk_level = model_service.get_risk_level(probability)

            predictions.append(
                PredictionOutput(
                    prediction=prediction,
                    probability=round(probability, 4),
                    risk_level=risk_level,
                    model_version=model_service.model_version,
                    timestamp=datetime.utcnow().isoformat()
                )
            )

        logger.info(f"Batch prediction completed: {len(predictions)} predictions")

        return BatchPredictionOutput(
            predictions=predictions,
            total_predictions=len(predictions),
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/explain", response_model=ExplainabilityOutput, tags=["Explainability"])
async def explain(input_data: PredictionInput):
    """
    Get SHAP explanation for a prediction

    Returns the prediction along with SHAP values showing feature contributions

    Note: This endpoint requires SHAP to be enabled in settings (ENABLE_SHAP=True)
    """
    try:
        if not settings.ENABLE_SHAP:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="SHAP explainability is disabled. Set ENABLE_SHAP=True in settings to enable this feature."
            )

        if not model_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model service is not ready"
            )

        if model_service.explainer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SHAP explainer is not initialized. Check logs for details."
            )

        # Get explanation
        explanation = model_service.explain_prediction(input_data.dict())

        logger.info(f"Explanation generated for prediction: {explanation['prediction']}")

        return ExplainabilityOutput(**explanation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
