# Heart Disease Prediction - MLOps Project

This project builds a machine learning system to predict heart disease using the UCI Heart Disease dataset. It includes model training, API deployment, Docker containers, and Kubernetes setup.

## What's Inside

This project covers the complete MLOps workflow:
- Data analysis and visualization
- Training multiple ML models (Random Forest, Logistic Regression, etc.)
- MLflow for tracking experiments
- FastAPI for serving predictions
- Docker containers
- Kubernetes deployment
- Monitoring with Prometheus and Grafana
- Automated tests

## Project Structure

```
MLOPS_Assignment1_2025/
├── app/                    # FastAPI application
├── src/                    # Training and preprocessing code
├── tests/                  # Test cases
├── notebooks/              # Jupyter notebooks for EDA
├── models/                 # Saved models
├── k8s/                    # Kubernetes configs
├── Dockerfile              
├── docker-compose.yml      
└── requirements.txt        
```

## Setup

**Requirements:**
- Python 3.9 or higher
- Docker (for containerization)
- Minikube (for Kubernetes)

**Install dependencies:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
{
  "prediction": 0,
  "probability": 0.234,
  "risk_level": "Low",
  "timestamp": "2025-12-30T10:30:00",
  "model_version": "1.0.0"
}
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

### Run Container

```bash
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
```

### Using Docker Compose

```bash
# Start all services (API, Prometheus, Grafana)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (Minikube, GKE, EKS, or AKS)
- kubectl configured

### Deploy to Kubernetes

```bash
# Load Docker image to Minikube (if using Minikube)
minikube image load heart-disease-api:latest

# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# Get service URL
kubectl get service heart-disease-api-service
```

### Access the API

```bash
# If using LoadBalancer
kubectl get service heart-disease-api-service

# If using Minikube
minikube service heart-disease-api-service
```

## 📈 Monitoring

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

Key metrics:
- API request rate
- Response latency
- Error rate
- Model prediction latency

### Grafana Dashboards

Access Grafana at `http://localhost:3000`

Default credentials: `admin/admin`

Pre-configured dashboards include:
- API Performance Dashboard
- Model Metrics Dashboard
- System Resources Dashboard

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v --cov=src --cov=app --cov-report=html
```

### Run Specific Test Suites

```bash
# Unit tests for preprocessing
pytest tests/test_preprocessing.py -v

# Unit tests for model
pytest tests/test_model.py -v

# Unit tests for API
pytest tests/test_api.py -v
```

### View Coverage Report

```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov=app --cov-report=html

# Open in browser
open htmlcov/index.html
```

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for CI/CD automation:

### Pipeline Stages

1. **Lint and Test**
   - Code formatting check (Black)
   - Import sorting check (isort)
   - Linting (Flake8)
   - Unit tests with coverage

2. **Build Docker**
   - Build Docker image
   - Run container health checks
   - Upload image artifact

3. **Train Model**
   - Run model training
   - Upload model artifacts

4. **Security Scan**
   - Vulnerability scanning with Trivy

### Trigger Pipeline

The pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

## 📝 Model Information

### Models Implemented

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble tree-based model
3. **Gradient Boosting** - Boosting ensemble model

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.85 | 0.86 | 0.84 | 0.85 | 0.92 |
| Random Forest | 0.87 | 0.88 | 0.86 | 0.87 | 0.94 |
| Random Forest (Tuned) | 0.89 | 0.90 | 0.88 | 0.89 | 0.95 |

### Features

- `age`: Age in years
- `sex`: Sex (1 = male; 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting ECG results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment
- `ca`: Number of major vessels (0-3)
- `thal`: Thalassemia type

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- Dr. Robert Detrano for the original data collection
- BITS Pilani for the MLOps course framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational purposes as part of the MLOps course at BITS Pilani.
