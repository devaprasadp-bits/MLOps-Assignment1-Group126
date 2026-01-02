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
- Multiple ML models (Random Forest worked best with 85% accuracy)
- MLflow for experiment tracking
- FastAPI for predictions
- Docker containers
- Kubernetes deployment on Minikube
- Monitoring with Prometheus and Grafana
- Tests with pytest (71% coverage)

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

## Running the Project

### 1. Data Analysis and Model Training

Run notebooks to explore data and train models:
```bash
jupyter notebook
```

View MLflow experiments:
```bash
mlflow ui
```
Then open http://localhost:5001

### 2. Run API Locally

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
  -H "Content-Type: application/json" \
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

- Dataset: UCI Heart Disease (303 samples, 14 features)
- Best model: Random Forest with ~85% accuracy  
- API: FastAPI with /health and /predict endpoints
- Tests: 25 tests, 71% coverage
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
