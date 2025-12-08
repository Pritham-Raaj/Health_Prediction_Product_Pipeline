"""
Simplified drift detection without Evidently AI
Uses statistical tests for distribution comparison
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


class DriftDetector:
    """Simple drift detector using statistical tests"""

    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None
        self.current_window: List[Dict] = []
        self.feature_names = [
            'AGE', 'CHOL', 'MAX_HEARTRATE', 'OLDPEAK', 'SEX_LABEL',
            'LOCATION_LABEL', 'CHESTPAINTYPE_LABEL', 'RESTINGBP_LABEL',
            'FASTINGBS_LABEL', 'RESTECG_LABEL', 'EXERCISE_CHESTPAIN_LABEL'
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
                self.reference_data = self._create_dummy_reference_data()
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            self.reference_data = self._create_dummy_reference_data()

    def _create_dummy_reference_data(self) -> pd.DataFrame:
        """Create dummy reference data for testing"""
        np.random.seed(42)
        data = {
            'AGE': np.random.randint(30, 80, 100),
            'CHOL': np.random.randint(150, 350, 100),
            'MAX_HEARTRATE': np.random.randint(100, 180, 100),
            'OLDPEAK': np.random.uniform(0, 5, 100),
            'SEX_LABEL': np.random.randint(0, 2, 100),
            'LOCATION_LABEL': np.random.randint(0, 4, 100),
            'CHESTPAINTYPE_LABEL': np.random.randint(0, 4, 100),
            'RESTINGBP_LABEL': np.random.randint(0, 3, 100),
            'FASTINGBS_LABEL': np.random.randint(0, 2, 100),
            'RESTECG_LABEL': np.random.randint(0, 3, 100),
            'EXERCISE_CHESTPAIN_LABEL': np.random.randint(0, 2, 100)
        }
        return pd.DataFrame(data)

    def add_prediction(self, input_data: Dict, prediction: int, probability: float):
        """Add a prediction to the monitoring window"""
        # Map input data to match feature names
        record = {
            'AGE': input_data.get('age', 0),
            'CHOL': input_data.get('cholesterol', 0),
            'MAX_HEARTRATE': input_data.get('max_heart_rate', 0),
            'OLDPEAK': input_data.get('oldpeak', 0.0),
            'SEX_LABEL': input_data.get('sex', 0),
            'LOCATION_LABEL': input_data.get('location', 0),
            'CHESTPAINTYPE_LABEL': input_data.get('chest_pain_type', 0),
            'RESTINGBP_LABEL': input_data.get('resting_bp', 0),
            'FASTINGBS_LABEL': input_data.get('fasting_bs', 0),
            'RESTECG_LABEL': input_data.get('resting_ecg', 0),
            'EXERCISE_CHESTPAIN_LABEL': input_data.get('exercise_angina', 0),
            'prediction': prediction,
            'probability': probability,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.current_window.append(record)

        # Keep window size manageable
        if len(self.current_window) > settings.MONITORING_WINDOW_SIZE:
            self.current_window.pop(0)

    def detect_drift(self) -> Dict:
        """Detect data drift using Kolmogorov-Smirnov test"""
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

            # Perform KS test for each feature
            drift_results = {}
            features_with_drift = []
            p_value_threshold = 0.05  # Significance level

            for feature in self.feature_names:
                if feature in current_data.columns and feature in self.reference_data.columns:
                    # Get reference and current distributions
                    ref_values = self.reference_data[feature].dropna()
                    curr_values = current_data[feature].dropna()

                    if len(ref_values) > 0 and len(curr_values) > 0:
                        # Perform Kolmogorov-Smirnov test
                        statistic, p_value = stats.ks_2samp(ref_values, curr_values)

                        drift_results[feature] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'drift_detected': p_value < p_value_threshold
                        }

                        if p_value < p_value_threshold:
                            features_with_drift.append(feature)

            # Overall drift detection
            drift_detected = len(features_with_drift) > 0
            drift_score = len(features_with_drift) / len(self.feature_names) if self.feature_names else 0

            return {
                "drift_detected": drift_detected,
                "drift_score": round(drift_score, 4),
                "features_with_drift": features_with_drift,
                "drift_details": drift_results,
                "current_window_size": len(self.current_window),
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Drift detection completed. {len(features_with_drift)} features showing drift." if drift_detected else "No significant drift detected."
            }

        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return {
                "drift_detected": False,
                "message": f"Error during drift detection: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    def get_data_quality_report(self) -> Dict:
        """Generate simple data quality report"""
        if len(self.current_window) < 5:
            return {
                "message": "Insufficient data for quality report",
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            current_data = pd.DataFrame(self.current_window)

            quality_metrics = {
                "total_records": len(current_data),
                "feature_stats": {}
            }

            for feature in self.feature_names:
                if feature in current_data.columns:
                    feature_data = current_data[feature]
                    quality_metrics["feature_stats"][feature] = {
                        "mean": float(feature_data.mean()),
                        "std": float(feature_data.std()),
                        "min": float(feature_data.min()),
                        "max": float(feature_data.max()),
                        "missing": int(feature_data.isna().sum())
                    }

            quality_metrics["timestamp"] = datetime.utcnow().isoformat()
            return quality_metrics

        except Exception as e:
            logger.error(f"Error in data quality report: {e}")
            return {
                "message": f"Error generating quality report: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    def save_reference_data(self, data: pd.DataFrame):
        """Save new reference data"""
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
