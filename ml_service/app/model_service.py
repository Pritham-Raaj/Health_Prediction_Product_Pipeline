"""
Model service for loading and serving predictions
"""
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from loguru import logger
import shap

from config.settings import settings


class ModelService:
    """Service for managing ML model and predictions"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = "1.0.0"
        self.explainer = None
        self._load_model()

    def _load_model(self):
        """Load the trained model, scaler, and feature names"""
        try:
            # Load model
            model_path = Path(settings.MODEL_PATH)
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}")

            # Load scaler
            scaler_path = Path(settings.SCALER_PATH)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded successfully from {scaler_path}")
            else:
                logger.warning(f"Scaler file not found at {scaler_path}")

            # Load feature names
            feature_names_path = Path(settings.FEATURE_NAMES_PATH)
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Feature names loaded: {self.feature_names}")
            else:
                logger.warning(f"Feature names file not found at {feature_names_path}")
                # Default feature names based on notebook
                self.feature_names = [
                    'AGE', 'SEX_LABEL', 'CHESTPAINTYPE_LABEL', 'RESTINGBP_LABEL',
                    'CHOL', 'FASTINGBS_LABEL', 'RESTECG_LABEL', 'MAX_HEARTRATE',
                    'EXERCISE_CHESTPAIN_LABEL', 'OLDPEAK'
                ]

            # Initialize SHAP explainer if model is loaded and SHAP is enabled
            if self.model is not None and settings.ENABLE_SHAP:
                try:
                    # For logistic regression, use LinearExplainer
                    self.explainer = shap.LinearExplainer(
                        self.model,
                        masker=shap.maskers.Independent(data=np.zeros((1, len(self.feature_names))))
                    )
                    logger.info("SHAP explainer initialized")
                except Exception as e:
                    logger.warning(f"SHAP explainer initialization failed: {e}. Explainability will be disabled.")
                    self.explainer = None

        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if model service is ready"""
        return self.model is not None and self.scaler is not None

    def preprocess_input(self, input_data: Dict) -> np.ndarray:
        """
        Preprocess input data to match training format

        Args:
            input_data: Dictionary with patient features

        Returns:
            Preprocessed numpy array ready for prediction
        """
        # Map input to feature array in the correct order as per feature_names.json:
        # ["AGE", "CHOL", "MAX_HEARTRATE", "OLDPEAK", "SEX_LABEL", "LOCATION_LABEL",
        #  "CHESTPAINTYPE_LABEL", "RESTINGBP_LABEL", "FASTINGBS_LABEL",
        #  "RESTECG_LABEL", "EXERCISE_CHESTPAIN_LABEL"]
        features = [
            input_data['age'],
            input_data['cholesterol'],
            input_data['max_heart_rate'],
            input_data['oldpeak'],
            input_data['sex'],
            input_data.get('location', 0),  # Default to 0 if not provided
            input_data['chest_pain_type'],
            input_data['resting_bp'],
            input_data['fasting_bs'],
            input_data['resting_ecg'],
            input_data['exercise_angina']
        ]

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Scale features
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)

        return features_array

    def predict(self, input_data: Dict) -> Tuple[int, float]:
        """
        Make prediction for a single input

        Args:
            input_data: Dictionary with patient features

        Returns:
            Tuple of (prediction, probability)
        """
        if not self.is_ready():
            raise RuntimeError("Model service is not ready")

        # Preprocess input
        features = self.preprocess_input(input_data)

        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]

        return int(prediction), float(probability)

    def predict_batch(self, inputs: List[Dict]) -> List[Tuple[int, float]]:
        """
        Make predictions for multiple inputs

        Args:
            inputs: List of input dictionaries

        Returns:
            List of (prediction, probability) tuples
        """
        results = []
        for input_data in inputs:
            prediction, probability = self.predict(input_data)
            results.append((prediction, probability))
        return results

    def explain_prediction(self, input_data: Dict) -> Dict:
        """
        Generate SHAP explanation for a prediction

        Args:
            input_data: Dictionary with patient features

        Returns:
            Dictionary with SHAP values and explanation
        """
        if not self.is_ready() or self.explainer is None:
            raise RuntimeError("Model service or explainer is not ready")

        # Preprocess input
        features = self.preprocess_input(input_data)

        # Get prediction
        prediction, probability = self.predict(input_data)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features)

        # Create feature importance dictionary
        shap_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            shap_dict[feature_name] = float(shap_values[0][i])

        # Get top contributing features
        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        top_features = [
            {"feature": str(feat), "contribution": float(contrib)}
            for feat, contrib in sorted_features
        ]

        return {
            "prediction": prediction,
            "probability": probability,
            "shap_values": shap_dict,
            "top_features": top_features,
            "base_value": float(self.explainer.expected_value)
        }

    def get_risk_level(self, probability: float) -> str:
        """
        Determine risk level based on probability

        Args:
            probability: Probability of heart disease

        Returns:
            Risk level as string
        """
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"


# Global model service instance
model_service = ModelService()
