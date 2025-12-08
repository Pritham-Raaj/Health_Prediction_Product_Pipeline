"""
Drift detection and monitoring using Evidently AI
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import *

from config.settings import settings


class DriftDetector:
    """Service for detecting data drift in predictions"""

    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None
        self.current_window: List[Dict] = []
        self.feature_names = [
            'AGE', 'SEX_LABEL', 'CHESTPAINTYPE_LABEL', 'RESTINGBP_LABEL',
            'CHOL', 'FASTINGBS_LABEL', 'RESTECG_LABEL', 'MAX_HEARTRATE',
            'EXERCISE_CHESTPAIN_LABEL', 'OLDPEAK'
        ]
        self._load_reference_data()

    def _load_reference_data(self):
        """Load reference data for drift detection"""
        try:
            reference_path = Path(settings.REFERENCE_DATA_PATH)
            if reference_path.exists():
                self.reference_data = pd.read_csv(reference_path)
                logger.info(f"Reference data loaded: {len(self.reference_data)} samples")
            else:
                logger.warning(f"Reference data not found at {reference_path}")
                # Create dummy reference data (will be replaced with actual data)
                self.reference_data = self._create_dummy_reference_data()
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            self.reference_data = self._create_dummy_reference_data()

    def _create_dummy_reference_data(self) -> pd.DataFrame:
        """Create dummy reference data for testing"""
        np.random.seed(42)
        data = {
            'AGE': np.random.randint(30, 80, 100),
            'SEX_LABEL': np.random.randint(0, 2, 100),
            'CHESTPAINTYPE_LABEL': np.random.randint(0, 4, 100),
            'RESTINGBP_LABEL': np.random.randint(0, 3, 100),
            'CHOL': np.random.randint(150, 350, 100),
            'FASTINGBS_LABEL': np.random.randint(0, 2, 100),
            'RESTECG_LABEL': np.random.randint(0, 3, 100),
            'MAX_HEARTRATE': np.random.randint(100, 180, 100),
            'EXERCISE_CHESTPAIN_LABEL': np.random.randint(0, 2, 100),
            'OLDPEAK': np.random.uniform(0, 5, 100)
        }
        return pd.DataFrame(data)

    def add_prediction(self, input_data: Dict, prediction: int, probability: float):
        """
        Add a prediction to the monitoring window

        Args:
            input_data: Input features
            prediction: Model prediction
            probability: Prediction probability
        """
        record = {
            'AGE': input_data['age'],
            'SEX_LABEL': input_data['sex'],
            'CHESTPAINTYPE_LABEL': input_data['chest_pain_type'],
            'RESTINGBP_LABEL': input_data['resting_bp'],
            'CHOL': input_data['cholesterol'],
            'FASTINGBS_LABEL': input_data['fasting_bs'],
            'RESTECG_LABEL': input_data['resting_ecg'],
            'MAX_HEARTRATE': input_data['max_heart_rate'],
            'EXERCISE_CHESTPAIN_LABEL': input_data['exercise_angina'],
            'OLDPEAK': input_data['oldpeak'],
            'prediction': prediction,
            'probability': probability,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.current_window.append(record)

        # Keep window size manageable
        if len(self.current_window) > settings.MONITORING_WINDOW_SIZE:
            self.current_window.pop(0)

    def detect_drift(self) -> Dict:
        """
        Detect data drift between reference and current data

        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            return {
                "drift_detected": False,
                "message": "Reference data not available",
                "timestamp": datetime.utcnow().isoformat()
            }

        if len(self.current_window) < 10:
            return {
                "drift_detected": False,
                "message": f"Insufficient data for drift detection (need at least 10 samples, have {len(self.current_window)})",
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            # Convert current window to DataFrame
            current_data = pd.DataFrame(self.current_window)
            current_features = current_data[self.feature_names]

            # Create drift report
            report = Report(metrics=[
                DataDriftPreset(),
            ])

            report.run(
                reference_data=self.reference_data[self.feature_names],
                current_data=current_features
            )

            # Extract drift results
            drift_results = report.as_dict()

            # Parse drift metrics
            drift_detected = False
            features_with_drift = []
            drift_score = 0.0

            try:
                metrics = drift_results.get('metrics', [])
                for metric in metrics:
                    if metric.get('metric') == 'DatasetDriftMetric':
                        result = metric.get('result', {})
                        drift_detected = result.get('dataset_drift', False)
                        drift_score = result.get('drift_share', 0.0)
                        drift_by_features = result.get('drift_by_columns', {})

                        for feature, drift_info in drift_by_features.items():
                            if isinstance(drift_info, dict) and drift_info.get('drift_detected', False):
                                features_with_drift.append(feature)
            except Exception as e:
                logger.warning(f"Error parsing drift metrics: {e}")

            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "features_with_drift": features_with_drift,
                "current_window_size": len(self.current_window),
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Drift analysis completed successfully"
            }

        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return {
                "drift_detected": False,
                "message": f"Error during drift detection: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    def get_data_quality_report(self) -> Dict:
        """
        Generate data quality report for current window

        Returns:
            Dictionary with data quality metrics
        """
        if len(self.current_window) < 5:
            return {
                "message": "Insufficient data for quality report",
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            current_data = pd.DataFrame(self.current_window)
            current_features = current_data[self.feature_names]

            # Create quality report
            report = Report(metrics=[
                DataQualityPreset(),
            ])

            report.run(
                reference_data=self.reference_data[self.feature_names],
                current_data=current_features
            )

            return {
                "report": report.as_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in data quality report: {e}")
            return {
                "message": f"Error generating quality report: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    def save_reference_data(self, data: pd.DataFrame):
        """
        Save new reference data

        Args:
            data: DataFrame to use as reference
        """
        try:
            reference_path = Path(settings.REFERENCE_DATA_PATH)
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(reference_path, index=False)
            self.reference_data = data
            logger.info(f"Reference data saved: {len(data)} samples")
        except Exception as e:
            logger.error(f"Error saving reference data: {e}")

    def clear_window(self):
        """Clear the current monitoring window"""
        self.current_window = []
        logger.info("Monitoring window cleared")


# Global drift detector instance
drift_detector = DriftDetector()
