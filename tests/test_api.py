"""
API tests for Heart Disease Prediction service
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_service"))

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()
        assert "version" in response.json()

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May be 503 if model not loaded
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestPredictionEndpoints:
    """Test prediction endpoints"""

    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        payload = {
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

        response = client.post("/predict", json=payload)

        # If model is loaded, should return 200
        # If not loaded, should return 503
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert data["prediction"] in [0, 1]
            assert 0.0 <= data["probability"] <= 1.0
            assert data["risk_level"] in ["Low", "Medium", "High"]

    def test_predict_invalid_age(self):
        """Test prediction with invalid age"""
        payload = {
            "age": 150,  # Invalid age
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

        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_missing_field(self):
        """Test prediction with missing required field"""
        payload = {
            "age": 55,
            "sex": 1,
            # Missing chest_pain_type
            "resting_bp": 140,
            "cholesterol": 250,
            "fasting_bs": 1,
            "resting_ecg": 0,
            "max_heart_rate": 150,
            "exercise_angina": 1,
            "oldpeak": 2.5
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_batch_predict(self):
        """Test batch predictions"""
        payload = {
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
                    "oldpeak": 2.5
                },
                {
                    "age": 48,
                    "sex": 0,
                    "chest_pain_type": 1,
                    "resting_bp": 120,
                    "cholesterol": 200,
                    "fasting_bs": 0,
                    "resting_ecg": 0,
                    "max_heart_rate": 160,
                    "exercise_angina": 0,
                    "oldpeak": 1.0
                }
            ]
        }

        response = client.post("/predict/batch", json=payload)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_predictions" in data
            assert len(data["predictions"]) == 2


class TestExplainabilityEndpoints:
    """Test explainability endpoints"""

    def test_explain(self):
        """Test SHAP explanation endpoint"""
        payload = {
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

        response = client.post("/explain", json=payload)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "shap_values" in data
            assert "top_features" in data
            assert "base_value" in data
            assert isinstance(data["shap_values"], dict)
            assert isinstance(data["top_features"], list)


class TestMonitoringEndpoints:
    """Test monitoring endpoints"""

    def test_drift_check(self):
        """Test drift detection endpoint"""
        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "message" in data
        assert "timestamp" in data

    def test_monitoring_stats(self):
        """Test monitoring statistics endpoint"""
        response = client.get("/monitoring/stats")
        assert response.status_code == 200
        data = response.json()
        assert "current_window_size" in data
        assert "drift_detection_enabled" in data
        assert "monitoring_enabled" in data

    def test_clear_window(self):
        """Test clear monitoring window endpoint"""
        response = client.post("/monitoring/clear")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
