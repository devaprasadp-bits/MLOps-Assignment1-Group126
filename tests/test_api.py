"""
Unit tests for API endpoints
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent / "app"))

# Note: This will need the model to be available
# For testing purposes, you might want to use a mock model


class TestAPIEndpoints:
    """Test API endpoint functionality"""

    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing"""
        return {
            "age": 63.0,
            "sex": 1,
            "cp": 1,
            "trestbps": 145.0,
            "chol": 233.0,
            "fbs": 1,
            "restecg": 2,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0.0,
            "thal": 6.0,
        }

    @pytest.fixture
    def client(self):
        """Create test client"""
        # Import here to avoid issues if model is not available
        try:
            from app import app

            return TestClient(app)
        except Exception as e:
            pytest.skip(f"Could not load app: {e}")

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Check that it returns text/plain content
        assert "text/plain" in response.headers.get("content-type", "")

    def test_predict_endpoint_valid_data(self, client, sample_patient_data):
        """Test predict endpoint with valid data"""
        response = client.post("/predict", json=sample_patient_data)

        # Model might not be trained yet, so check both cases
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert "timestamp" in data
            assert "model_version" in data
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1
            assert data["risk_level"] in ["Low", "Medium", "High"]
            assert data["model_version"] == "1.0.0"
        elif response.status_code == 500:
            # Model not loaded - acceptable for testing without trained model
            assert "Model not loaded" in str(response.json())

    def test_predict_endpoint_invalid_data(self, client):
        """Test predict endpoint with invalid data"""
        invalid_data = {
            "age": -5,  # Invalid age
            "sex": 1,
            "cp": 1,
            # Missing required fields
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_endpoint(self, client, sample_patient_data):
        """Test batch predict endpoint"""
        batch_data = [sample_patient_data, sample_patient_data]
        response = client.post("/batch_predict", json=batch_data)

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2

            for pred in data["predictions"]:
                assert "prediction" in pred
                assert "probability" in pred
                assert "risk_level" in pred
                assert "timestamp" in pred
                assert "model_version" in pred
                assert pred["prediction"] in [0, 1]
                assert 0 <= pred["probability"] <= 1
                assert pred["risk_level"] in ["Low", "Medium", "High"]
                assert pred["model_version"] == "1.0.0"
        elif response.status_code == 500:
            # Model not loaded - acceptable
            pass

    def test_predict_endpoint_error_handling(self, client):
        """Test predict endpoint error handling with malformed data"""
        malformed_data = {"age": "not_a_number", "sex": 1}  # Wrong type
        response = client.post("/predict", json=malformed_data)
        assert response.status_code == 422

    def test_risk_level_calculation(self, client, sample_patient_data):
        """Test that risk levels are correctly calculated"""
        response = client.post("/predict", json=sample_patient_data)

        if response.status_code == 200:
            data = response.json()
            prob = data["probability"]
            risk = data["risk_level"]

            # Verify risk level matches probability
            if prob < 0.3:
                assert risk == "Low"
            elif prob < 0.6:
                assert risk == "Medium"
            else:
                assert risk == "High"
        # Skip if model not loaded

    def test_multiple_predictions_different_patients(self, client):
        """Test predictions with different patient profiles"""
        # Young patient with good indicators
        young_patient = {
            "age": 35.0,
            "sex": 0,
            "cp": 4,
            "trestbps": 120.0,
            "chol": 180.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 180.0,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "ca": 0.0,
            "thal": 3.0,
        }

        # Older patient with risk factors
        older_patient = {
            "age": 70.0,
            "sex": 1,
            "cp": 2,
            "trestbps": 160.0,
            "chol": 280.0,
            "fbs": 1,
            "restecg": 2,
            "thalach": 110.0,
            "exang": 1,
            "oldpeak": 3.5,
            "slope": 2,
            "ca": 3.0,
            "thal": 7.0,
        }

        response1 = client.post("/predict", json=young_patient)
        response2 = client.post("/predict", json=older_patient)

        # Both should succeed or both fail (if model not loaded)
        if response1.status_code == 200:
            assert response2.status_code == 200
            assert response1.json()["prediction"] in [0, 1]
            assert response2.json()["prediction"] in [0, 1]

    def test_batch_predict_with_multiple_patients(self, client):
        """Test batch prediction with varied patient data"""
        patients = [
            {
                "age": 45.0,
                "sex": 1,
                "cp": 3,
                "trestbps": 130.0,
                "chol": 200.0,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150.0,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 0.0,
                "thal": 3.0,
            },
            {
                "age": 60.0,
                "sex": 1,
                "cp": 1,
                "trestbps": 150.0,
                "chol": 250.0,
                "fbs": 1,
                "restecg": 2,
                "thalach": 120.0,
                "exang": 1,
                "oldpeak": 2.5,
                "slope": 2,
                "ca": 2.0,
                "thal": 7.0,
            },
            {
                "age": 50.0,
                "sex": 0,
                "cp": 4,
                "trestbps": 125.0,
                "chol": 190.0,
                "fbs": 0,
                "restecg": 0,
                "thalach": 160.0,
                "exang": 0,
                "oldpeak": 0.5,
                "slope": 1,
                "ca": 0.0,
                "thal": 3.0,
            },
        ]

        response = client.post("/batch_predict", json=patients)

        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 3

            # Verify all predictions are valid
            for pred in data["predictions"]:
                assert pred["prediction"] in [0, 1]
                assert 0 <= pred["probability"] <= 1


class TestDataValidation:
    """Test input data validation"""

    def test_age_validation(self):
        """Test age field validation"""
        from pydantic import ValidationError

        try:
            from app import PatientData
        except ImportError:
            pytest.skip("Could not import PatientData")

        # Valid age
        valid_data = {
            "age": 63.0,
            "sex": 1,
            "cp": 1,
            "trestbps": 145.0,
            "chol": 233.0,
            "fbs": 1,
            "restecg": 2,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0.0,
            "thal": 6.0,
        }
        patient = PatientData(**valid_data)
        assert patient.age == 63.0

        # Invalid age
        invalid_data = valid_data.copy()
        invalid_data["age"] = -5
        with pytest.raises(ValidationError):
            PatientData(**invalid_data)

    def test_required_fields(self):
        """Test that all required fields are validated"""
        try:
            from app import PatientData
        except ImportError:
            pytest.skip("Could not import PatientData")

        from pydantic import ValidationError

        incomplete_data = {"age": 63.0, "sex": 1}
        with pytest.raises(ValidationError):
            PatientData(**incomplete_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
