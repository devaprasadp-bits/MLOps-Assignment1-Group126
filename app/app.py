"""
Heart Disease Prediction API
FastAPI application for serving heart disease predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import List
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk based on patient health data",
    version="1.0.0",
)

# Prometheus metrics
prediction_counter = Counter("predictions_total", "Total number of predictions")
prediction_errors = Counter("prediction_errors_total", "Total number of prediction errors")
prediction_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

# Load model at startup
MODEL_PATH = Path("models/best_model.pkl")
model = None


class PatientData(BaseModel):
    """Input schema for patient data"""

    age: float = Field(..., description="Age in years", ge=0, le=120)
    sex: int = Field(..., description="Sex (1=male, 0=female)", ge=0, le=1)
    cp: int = Field(..., description="Chest pain type (0-3)", ge=0, le=4)
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)", ge=0)
    chol: float = Field(..., description="Serum cholesterol (mg/dl)", ge=0)
    fbs: int = Field(
        ...,
        description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)",
        ge=0,
        le=1,
    )
    restecg: int = Field(..., description="Resting ECG results (0-2)", ge=0, le=2)
    thalach: float = Field(..., description="Maximum heart rate achieved", ge=0, le=250)
    exang: int = Field(..., description="Exercise induced angina (1=yes, 0=no)", ge=0, le=1)
    oldpeak: float = Field(..., description="ST depression induced by exercise", ge=0)
    slope: int = Field(..., description="Slope of peak exercise ST segment (1-3)", ge=0, le=3)
    ca: float = Field(..., description="Number of major vessels (0-3)", ge=0, le=4)
    thal: float = Field(
        ...,
        description="Thalassemia (3=normal, 6=fixed, 7=reversible)",
        ge=0,
        le=7,
    )

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Output schema for predictions"""

    prediction: int
    probability: float
    risk_level: str
    timestamp: str
    model_version: str


@app.on_event("startup")
async def load_model():
    """Load the ML model on startup"""
    global model
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics",
            "/docs": "GET - API documentation",
        },
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Make a heart disease prediction

    Args:
        patient: Patient health data

    Returns:
        Prediction with probability and risk level
    """
    try:
        # Track metrics
        prediction_counter.inc()

        # Log request
        logger.info(f"Prediction request received: {patient.dict()}")

        # Check if model is loaded
        if model is None:
            prediction_errors.inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert input to DataFrame
        input_data = pd.DataFrame([patient.dict()])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Prepare response
        response = PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0",
        )

        # Log response
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Risk: {risk_level}")

        return response

    except Exception as e:
        prediction_errors.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(patients: List[PatientData]):
    """
    Make predictions for multiple patients

    Args:
        patients: List of patient health data

    Returns:
        List of predictions
    """
    try:
        logger.info(f"Batch prediction request received for {len(patients)} patients")

        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert inputs to DataFrame
        input_data = pd.DataFrame([p.dict() for p in patients])

        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        # Prepare responses
        responses = []
        for pred, prob in zip(predictions, probabilities):
            if prob < 0.3:
                risk_level = "Low"
            elif prob < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"

            responses.append(
                {
                    "prediction": int(pred),
                    "probability": float(prob),
                    "risk_level": risk_level,
                    "timestamp": datetime.now().isoformat(),
                    "model_version": "1.0.0",
                }
            )

        logger.info(f"Batch prediction completed for {len(patients)} patients")
        return {"predictions": responses}

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
