# Heart Disease Prediction - MLOps Project

**GitHub Repository:** https://github.com/devaprasadp-bits/MLOps-Assignment1-Group126

**Group 126** - MLOps Assignment 1 (S1-25_AIMLCZG523)

## Contributors:
- Devaprasad P           (2023aa05069@wilp.bits-pilani.ac.in)
- Devender Kumar         (2024aa05065@wilp.bits-pilani.ac.in)
- Chavali Amrutha Valli  (2024aa05610@wilp.bits-pilani.ac.in)
- Palakolanu Preethi     (2024aa05608@wilp.bits-pilani.ac.in)
- Rohan Tirthankar Behera(2024aa05607@wilp.bits-pilani.ac.in)

This project builds a machine learning system to predict heart disease as part of an MLOps coursework assignment using the UCI Heart Disease dataset. It covers data analysis, model training, API deployment, and the steps needed to run this end to end with Docker and Kubernetes.

## What's Included

- Data analysis with Jupyter notebooks
- Multiple ML models (Random Forest Tuned achieved best performance with 90.16% accuracy and 95.67% ROC-AUC)
- MLflow for experiment tracking
- FastAPI for predictions
- Docker containers
- Kubernetes deployment on Minikube
- Monitoring with Prometheus and Grafana
- Tests with pytest (72.84% coverage, 26 tests)

## Project Structure

```
MLOPS_Assignment1_2025/
├── app/                    # FastAPI application
├── src/                    # Training and preprocessing code
├── tests/                  # Test cases
├── notebooks/              # Jupyter notebooks for EDA
├── models/                 # Saved models
├── k8s/                    # Kubernetes configs
├── screenshots/            # Phase screenshots
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile              
├── docker-compose.yml      
├── prometheus.yml          # Prometheus configuration
├── setup.cfg               # Test configuration
└── requirements.txt        
```

## Prerequisites

- Python 3.9 or 3.10
- Docker and Docker Compose
- Git
- (Optional) Minikube and kubectl for Kubernetes deployment
- (Optional) Jupyter for running notebooks

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/devaprasadp-bits/MLOps-Assignment1-Group126.git
cd MLOps-Assignment1-Group126

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Prepare data
python scripts/prepare_data.py
```

### 2. Train Model (Required)

**Note:** Models are not included in the repository. You must train a model before running the API.

**Option A - Using Script (Recommended for quick setup):**
```bash
# Train all models with hyperparameter tuning
python scripts/train_model.py --model all --tune

# Or use the Makefile
make train
```

**Option B - Using Notebook (Recommended for learning/exploration):**
```bash
# Open the model development notebook
jupyter notebook notebooks/02_model_development.ipynb
# Run all cells to train and save the best model
```

Both options will:
- Load preprocessed data from `data/processed/heart_disease_clean.csv`
- Train multiple models (Logistic Regression, Random Forest, Gradient Boosting)
- Perform hyperparameter tuning
- Log experiments to MLflow
- Save the best model to `models/best_model.pkl`

### 3. View MLflow Experiments

```bash
mlflow ui --port 5001
```
Then open http://localhost:5001

## Running the Project

## Running the Project

After completing the Quick Start setup above, you can explore the project:

### Optional: Data Analysis

Run notebooks to explore data:
```bash
jupyter notebook
# Open notebooks/01_data_acquisition_and_eda.ipynb
# Or notebooks/02_model_development.ipynb
```

### 4. Run API Locally

**Prerequisites:** Complete step 2 (Train Model) first to create `models/best_model.pkl`.

Start the API:
```bash
uvicorn app.app:app --reload --port 8000
```

Test it:
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H5. Docker

**Prerequisites:** Complete step 2 (Train Model) first.-Type: application/json" \
  -d '{"age": 63.0, "sex": 1, "cp": 1, "trestbps": 145.0, "chol": 233.0, "fbs": 1, "restecg": 2, "thalach": 150.0, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0.0, "thal": 6.0}'
```

### 3. Docker

Build and run:
```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

Or use docker-compose (runs API + Prometheus + Grafana):
```bash
docker-compose up
```
6. Kubernetes

**Prerequisites:** Complete step 2 (Train Model) and step 5 (Docker build) first.
Access:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login: admin/admin)

### 4. Kubernetes

Deploy to Minikube:
```bash
minikube start
minikube image load heart-disease-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl get pods
```

Get service URL:
```bash
minikube service heart-disease-api-service --url
```

Test scaling:
```bash
kubectl scale deployment heart-disease-api --replicas=5
kubectl get pods
```

## Testing

Run tests:
```bash
pytest --cov=src --cov=app tests/
```

Test results: 72.84% coverage with 26 tests passing.

## Problems We Faced

1. **NumPy version issue**: Model trained with NumPy 2.x but Docker used 1.24.3. Fixed by changing requirements.txt to `numpy>=1.24.3`.

2. **Missing /metrics endpoint**: Prometheus was getting 404s. Had to add prometheus-client and create the endpoint.

3. **Docker authentication**: Got 401 errors pushing to Docker Hub. Needed to run `docker login` first.

4. **Jupyter catching commands**: Jupyter server was intercepting terminal commands. Had to use separate tabs.

## Results
Models trained: Logistic Regression, Random Forest, Gradient Boosting, Random Forest Tuned
- Best model: Random Forest Tuned with **90.16% test accuracy** and **95.67% ROC-AUC**
- API: FastAPI with /health, /predict, /metrics, and docs endpoints
- Tests: 26 tests passing, 72.84% coverage
- Deployment: Kubernetes with scalable replicas
- Monitoring: Prometheus + Grafana tracking predictions and system metric
- Deployment: Kubernetes with 5 replicas
- Monitoring: Prometheus + Grafana tracking predictions

## Notes

- Make sure Docker is running before docker-compose
- Minikube needs to be started before kubectl commands
- mlruns/ folder contains all MLflow experiment data
- Screenshots are in screenshots/ folder
- Model training is run manually and is not part of the CI pipeline to avoid long CI runtimes

---

MLOps Assignment - January 2026
