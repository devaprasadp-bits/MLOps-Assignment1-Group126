"""
Unit tests for model training and prediction
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


class TestModelTraining:
    """Test model training functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "age": np.random.randint(30, 80, 100),
                "sex": np.random.randint(0, 2, 100),
                "cp": np.random.randint(0, 4, 100),
                "trestbps": np.random.randint(90, 200, 100),
                "chol": np.random.randint(120, 400, 100),
                "fbs": np.random.randint(0, 2, 100),
                "restecg": np.random.randint(0, 3, 100),
                "thalach": np.random.randint(70, 200, 100),
                "exang": np.random.randint(0, 2, 100),
                "oldpeak": np.random.uniform(0, 6, 100),
                "slope": np.random.randint(1, 4, 100),
                "ca": np.random.randint(0, 4, 100),
                "thal": np.random.choice([3, 6, 7], 100),
            }
        )
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_logistic_regression_training(self, sample_data):
        """Test Logistic Regression model training"""
        X, y = sample_data

        model = Pipeline(
            [("scaler", StandardScaler()), ("classifier", LogisticRegression(random_state=42))]
        )

        model.fit(X, y)

        # Assertions
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)

    def test_random_forest_training(self, sample_data):
        """Test Random Forest model training"""
        X, y = sample_data

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        model.fit(X, y)

        # Assertions
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_model_predictions_shape(self, sample_data):
        """Test prediction output shapes"""
        X, y = sample_data

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        model.fit(X, y)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        assert predictions.shape == (len(X),)
        assert probabilities.shape == (len(X), 2)

    def test_model_save_load(self, sample_data, tmp_path):
        """Test model saving and loading"""
        X, y = sample_data

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        model.fit(X, y)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        # Load model
        loaded_model = joblib.load(model_path)

        # Test predictions match
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)

        assert np.array_equal(original_pred, loaded_pred)


class TestModelPredictions:
    """Test model prediction functionality"""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing"""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "age": np.random.randint(30, 80, 100),
                "sex": np.random.randint(0, 2, 100),
                "cp": np.random.randint(0, 4, 100),
                "trestbps": np.random.randint(90, 200, 100),
                "chol": np.random.randint(120, 400, 100),
                "fbs": np.random.randint(0, 2, 100),
                "restecg": np.random.randint(0, 3, 100),
                "thalach": np.random.randint(70, 200, 100),
                "exang": np.random.randint(0, 2, 100),
                "oldpeak": np.random.uniform(0, 6, 100),
                "slope": np.random.randint(1, 4, 100),
                "ca": np.random.randint(0, 4, 100),
                "thal": np.random.choice([3, 6, 7], 100),
            }
        )
        y = np.random.randint(0, 2, 100)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )
        model.fit(X, y)

        return model

    def test_single_prediction(self, trained_model):
        """Test single prediction"""
        input_data = pd.DataFrame(
            [
                {
                    "age": 63,
                    "sex": 1,
                    "cp": 1,
                    "trestbps": 145,
                    "chol": 233,
                    "fbs": 1,
                    "restecg": 2,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 3,
                    "ca": 0,
                    "thal": 6,
                }
            ]
        )

        prediction = trained_model.predict(input_data)
        probability = trained_model.predict_proba(input_data)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
        assert 0 <= probability[0][1] <= 1

    def test_batch_prediction(self, trained_model):
        """Test batch predictions"""
        input_data = pd.DataFrame(
            [
                {
                    "age": 63,
                    "sex": 1,
                    "cp": 1,
                    "trestbps": 145,
                    "chol": 233,
                    "fbs": 1,
                    "restecg": 2,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 3,
                    "ca": 0,
                    "thal": 6,
                },
                {
                    "age": 67,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 160,
                    "chol": 286,
                    "fbs": 0,
                    "restecg": 2,
                    "thalach": 108,
                    "exang": 1,
                    "oldpeak": 1.5,
                    "slope": 2,
                    "ca": 3,
                    "thal": 3,
                },
            ]
        )

        predictions = trained_model.predict(input_data)
        probabilities = trained_model.predict_proba(input_data)

        assert len(predictions) == 2
        assert len(probabilities) == 2
        assert all(p in [0, 1] for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
